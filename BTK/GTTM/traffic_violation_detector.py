import os
import time
import cv2
import numpy as np
import traceback
from ultralytics import YOLO

class ViolationDetector:
    def __init__(self, video_source):
        self.line_pts = []
        self.select_line(video_source)
        self.car_model = YOLO('yolov8n.pt')
        self.tl_model = YOLO('C:/Users/hi/Downloads/Smart-City-and-Smart-Agriculture-main/Smart_City-Case_study/best_traffic_nano_yolo.pt')
        self.track_history = {}
        self.violations = []  # <-- thêm để lưu danh sách vi phạm
        self.frame_count = 0
        self.cap = cv2.VideoCapture(video_source)
        os.makedirs('vi_pham', exist_ok=True)

    def select_line(self, video_source):
        cap = cv2.VideoCapture(video_source)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Không thể đọc video.")
        cv2.namedWindow("Select Line")
        cv2.setMouseCallback("Select Line", self.mouse_callback)
        while True:
            disp = first_frame.copy()
            if len(self.line_pts) >= 1:
                cv2.circle(disp, self.line_pts[0], 5, (0,255,0), -1)
            if len(self.line_pts) == 2:
                cv2.line(disp, self.line_pts[0], self.line_pts[1], (0,255,0), 2)
            cv2.imshow("Select Line", disp)
            key = cv2.waitKey(1) & 0xFF
            if len(self.line_pts) == 2 and key == ord('s'):
                break
            if key == ord('q'):
                cap.release(); cv2.destroyAllWindows(); exit("Hủy chọn.")
        cv2.destroyWindow("Select Line")
        cap.release()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.line_pts) < 2:
            self.line_pts.append((x, y))
            print(f"Point {len(self.line_pts)}: {(x,y)}")

    def side_of_line(self, pt, p1, p2):
        x1,y1 = p1; x2,y2 = p2
        return (x2-x1)*(pt[1]-y1) - (y2-y1)*(pt[0]-x1)

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        try:
            self.frame_count += 1
            orig_frame = frame.copy()

            # Vẽ vạch
            cv2.line(frame, self.line_pts[0], self.line_pts[1], (0,255,0), 2)

            # Detect đèn giao thông
            tl_res = self.tl_model(frame, conf=0.3)[0]
            tl_state = None
            for tl_box in tl_res.boxes:
                x1_l,y1_l,x2_l,y2_l = tl_box.xyxy.cpu().numpy().astype(int)[0]
                cls_id = int(tl_box.cls.cpu().item())
                conf_l = float(tl_box.conf.cpu().item())
                name = self.tl_model.model.names[cls_id]
                color = (0,255,0) if name=='green' else (0,0,255) if name=='red' else (255,255,0)
                cv2.rectangle(frame, (x1_l,y1_l),(x2_l,y2_l), color, 2)
                cv2.putText(frame, f"{name}:{conf_l:.2f}", (x1_l,y1_l-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if tl_state is None or conf_l > tl_state[1]:
                    tl_state = (name, conf_l)

            light_label = tl_state[0] if tl_state else "no-light"
            cv2.putText(frame, f"Light: {light_label}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0) if light_label=='green' else (0,0,255), 2)

            # Tracking
            car_results = self.car_model.track(frame, conf=0.5, iou=0.5, persist=True, stream=False)
            for box in car_results[0].boxes:
                if box.id is None:
                    continue
                tid = int(box.id.cpu().item())
                x1,y1,x2,y2 = box.xyxy.cpu().numpy().astype(int)[0]
                cx = (x1 + x2)//2
                cy = y2

                if tid not in self.track_history:
                    self.track_history[tid] = {
                        'pt': (cx,cy),
                        'crossed': False,
                        'violation': False,
                        'violation_time': None
                    }
                rec = self.track_history[tid]

                box_color = (0,0,255) if rec['violation'] else (255,0,0)

                if not rec['crossed']:
                    s_prev = self.side_of_line(rec['pt'], self.line_pts[0], self.line_pts[1])
                    s_curr = self.side_of_line((cx,cy), self.line_pts[0], self.line_pts[1])
                    if s_prev * s_curr < 0:
                        if light_label == 'red':
                            rec['violation'] = True
                            rec['violation_time'] = time.time()
                            crop = orig_frame[y1:y2, x1:x2]
                            fname = os.path.join('vi_pham', f"car_{tid}_{self.frame_count}.jpg")
                            cv2.imwrite(fname, crop)
                            print(f"[VI PHAM] Saved {fname}")
                            # ➕ Ghi nhận vào danh sách
                            self.violations.append({
                                'id': tid,
                                'timestamp': time.strftime("%H:%M:%S"),
                                'image': fname
                            })
                        rec['crossed'] = True

                rec['pt'] = (cx,cy)

                cv2.rectangle(frame, (x1,y1),(x2,y2), box_color, 2)
                cv2.putText(frame, f"ID:{tid}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                cv2.circle(frame, (cx,cy), 4, box_color, -1)

                if rec['violation'] and rec['violation_time'] is not None:
                    if time.time() - rec['violation_time'] <= 1.0:
                        cv2.putText(frame, "VI PHAM", (x1, y1-30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            violation_count = sum(1 for v in self.track_history.values() if v['violation'])
            cv2.putText(frame, f"Violations: {violation_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            return frame

        except Exception as e:
            print(f"[WARNING] Error during frame processing: {e}")
            traceback.print_exc()
            return frame

    def get_violations(self):
        """Hàm trả về danh sách vi phạm cho giao diện web"""
        return self.violations