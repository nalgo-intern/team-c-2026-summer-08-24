import os
import tempfile
import time
import cv2
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from moviepy import VideoFileClip

from squat_counter import SquatCounter
from judge_squat import SquatEvaluator
from pushup_counter import PushupCounter
from judge_pushup import PushupEvaluator
from exercise_detector import ExerciseDetector

# ==========================================
# 1. Logic層（動画の保存や解析処理）
# ==========================================
class VideoProcessor:
    def __init__(self):
        pass

    def process_and_render(self, uploaded_file):
        if uploaded_file is None:
            return None, None, None

        # ここで毎回新しくインスタンス化し、タイムスタンプや履歴をリセットする
        detector = ExerciseDetector(
            model_path="models/lstm_binary_pushup_squat.pth", 
            pose_model_path="pose_landmarker_lite.task"
        )

        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        input_temp.write(uploaded_file.read())
        input_temp.close()

        cap = cv2.VideoCapture(input_temp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))

        squat_counter = SquatCounter()
        squat_eval = SquatEvaluator()
        pushup_counter = PushupCounter()
        pushup_eval = PushupEvaluator()

        frame_index = 0
        last_valid_ex_code = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp_ms = (frame_index / fps) * 1000.0
            current_second = frame_index / fps
            
            ex_code = detector.process_frame(frame, timestamp_ms)
            
            if ex_code in (1, 2):
                last_valid_ex_code = ex_code
            else:
                ex_code = last_valid_ex_code

            landmarks = detector.pose_estimator.current_landmarks
            angles = detector.pose_estimator.records[-1] if detector.pose_estimator.records else {}
            
            annotated_frame = detector.pose_estimator.draw_landmarks(frame.copy())
            
            current_ex = "Unknown"
            count_val = 0
            eval_val = 0.0
            eval_key = ""

            if ex_code == 1: # Push-up
                current_ex = "Push-up"
                pushup_counter.update_from_pose_angles(angles)
                count_val = pushup_counter.count
                if landmarks:
                    res = pushup_eval.judge_frame(current_second, landmarks)
                    if res:
                        eval_val = res.get("shoulder_angle", 0.0)
                        eval_key = "Shoulder Angle"
            elif ex_code == 2: # Squat
                current_ex = "Squat"
                squat_counter.update_from_pose_angles(angles)
                count_val = squat_counter.count
                if landmarks:
                    res = squat_eval.judge_frame(current_second, landmarks)
                    if res:
                        eval_val = res.get("knee_angle", 0.0)
                        eval_key = "Knee Angle"
            
            # テキストの描画
            cv2.putText(annotated_frame, f"Exercise: {current_ex}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Count: {count_val}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            if eval_key:
                cv2.putText(annotated_frame, f"{eval_key}: {eval_val:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            out.write(annotated_frame)
            frame_index += 1

        cap.release()
        out.release()
        
        # H.264への変換 (Streamlitでの表示用)
        final_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        clip = VideoFileClip(output_temp.name)
        clip.write_videofile(final_output, codec="libx264", audio=False, logger=None)
        clip.close()
        
        os.remove(input_temp.name)
        os.remove(output_temp.name)

        df_angles = detector.pose_estimator.get_dataframe()
        results = {
            "squat": {"count": squat_counter.count, "eval": squat_eval.get_result()},
            "pushup": {"count": pushup_counter.count, "eval": pushup_eval.get_result()}
        }

        return final_output, df_angles, results


# ==========================================
# WebRTC用の映像処理クラス
# ==========================================
class RealtimeVideoProcessor:
    def __init__(self):
        self.detector = ExerciseDetector(
            model_path="models/lstm_binary_pushup_squat.pth", 
            pose_model_path="pose_landmarker_lite.task"
        )
        self.squat_counter = SquatCounter()
        self.squat_eval = SquatEvaluator()
        self.pushup_counter = PushupCounter()
        self.pushup_eval = PushupEvaluator()
        
        self.start_time = time.time()
        
        self.current_exercise = "WAITING"
        self.current_count = 0
        self.current_stage = "WAITING"
        self.feedback_msg = None
        
        self.last_valid_ex_code = 0 
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        current_time = time.time()
        elapsed_seconds = current_time - self.start_time
        timestamp_ms = int(elapsed_seconds * 1000)

        ex_code = self.detector.process_frame(img, timestamp_ms)

        if ex_code in (1, 2):
            self.last_valid_ex_code = ex_code
        else:
            ex_code = self.last_valid_ex_code

        landmarks = self.detector.pose_estimator.current_landmarks
        angles = self.detector.pose_estimator.records[-1] if self.detector.pose_estimator.records else {}

        annotated_frame = self.detector.pose_estimator.draw_landmarks(img.copy())

        if ex_code == 1:
            self.current_exercise = "Push-up"
            self.current_stage = self.pushup_counter.update_from_pose_angles(angles)
            self.current_count = self.pushup_counter.count
            if landmarks:
                self.pushup_eval.judge_frame(elapsed_seconds, landmarks)
                if self.pushup_eval.feedback_logs:
                    latest = self.pushup_eval.feedback_logs[-1]
                    if elapsed_seconds - latest["time"] < 3.0:
                        self.feedback_msg = latest["message"]
                        
        elif ex_code == 2:
            self.current_exercise = "Squat"
            self.current_stage = self.squat_counter.update_from_pose_angles(angles)
            self.current_count = self.squat_counter.count
            if landmarks:
                self.squat_eval.judge_frame(elapsed_seconds, landmarks)
                if self.squat_eval.feedback_logs:
                    latest = self.squat_eval.feedback_logs[-1]
                    if elapsed_seconds - latest["time"] < 3.0:
                        self.feedback_msg = latest["message"]
        else:
            self.current_exercise = "Unknown"

        cv2.putText(annotated_frame, f"Exercise: {self.current_exercise}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Count: {self.current_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# ==========================================
# 2. State層（状態管理）
# ==========================================
class SessionManager:
    @staticmethod
    def init_state():
        if "processed_video_path" not in st.session_state:
            st.session_state.processed_video_path = None
        if "processed_df" not in st.session_state:
            st.session_state.processed_df = None
        if "results" not in st.session_state:
            st.session_state.results = None

    @staticmethod
    def get(key: str):
        return st.session_state.get(key)

    @staticmethod
    def set(key: str, value):
        st.session_state[key] = value


# ==========================================
# 3. View層（画面の描画）
# ==========================================
class MainPageView:
    def __init__(self, processor: VideoProcessor):
        self.processor = processor

    def render(self):
        st.title("筋トレアプリ")
        
        tab1, tab2 = st.tabs(["動画", "カメラ"])
        
        with tab1:
            self._render_upload_tab()
            
        with tab2:
            self._render_realtime_tab()

    def _render_upload_tab(self):
        uploaded_video = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "mov", "avi"])

        if uploaded_video is not None:
            st.write("元動画")
            st.video(uploaded_video)

            if st.button("解析実行"):
                with st.spinner("解析中"):
                    out_vid, out_df, results = self.processor.process_and_render(uploaded_video)
                    
                    SessionManager.set("processed_video_path", out_vid)
                    SessionManager.set("processed_df", out_df)
                    SessionManager.set("results", results)

        processed_vid = SessionManager.get("processed_video_path")
        processed_df = SessionManager.get("processed_df")
        results = SessionManager.get("results")

        if processed_vid and processed_df is not None and results:
            st.success("解析が完了しました")

            col1, col2 = st.columns(2)
            
            # --- スクワットの結果 ---
            with col1:
                st.subheader("スクワット")
                sq_data = results["squat"]
                st.metric(label="回数", value=f"{sq_data['count']}回")
                st.write(sq_data["eval"]["summary"])
                if sq_data["eval"]["feedback_logs"]:
                    with st.expander("フィードバック詳細 (スクワット)"):
                        for log in sq_data["eval"]["feedback_logs"]:
                            st.write(f"- {log['time']:.1f}秒: {log['message']}")
                            
            # --- 腕立ての結果 ---
            with col2:
                st.subheader("腕立て伏せ")
                pu_data = results["pushup"]
                st.metric(label="回数", value=f"{pu_data['count']}回")
                st.write(pu_data["eval"]["summary"])
                if pu_data["eval"]["feedback_logs"]:
                    with st.expander("フィードバック詳細 (腕立て伏せ)"):
                        for log in pu_data["eval"]["feedback_logs"]:
                            st.write(f"- {log['time']:.1f}秒: {log['message']}")
            
            st.divider()
            
            vid_col, df_col = st.columns([1, 1])
            
            with vid_col:
                st.write("解析結果動画")
                with open(processed_vid, 'rb') as f:
                    st.video(f.read())
            
            with df_col:
                st.write("各フレームの関節角度データ")
                st.dataframe(processed_df)
                
                csv = processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="CSVとしてダウンロード",
                    data=csv,
                    file_name='pose_angles.csv',
                    mime='text/csv',
                )

    def _render_realtime_tab(self):
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        video_col, metrics_col = st.columns([2, 1])

        with video_col:
            webrtc_ctx = webrtc_streamer(
                key="workout-realtime",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=RealtimeVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with metrics_col:
            st.markdown("### ステータス")
            ex_placeholder = st.empty()
            count_placeholder = st.empty()
            stage_placeholder = st.empty()
            feedback_placeholder = st.empty()
            
            if webrtc_ctx.state.playing:
                while True:
                    if webrtc_ctx.video_processor:
                        ex_placeholder.metric("種目", webrtc_ctx.video_processor.current_exercise)
                        count_placeholder.metric("回数", f"{webrtc_ctx.video_processor.current_count} 回")
                        stage_placeholder.metric("現在の状態", webrtc_ctx.video_processor.current_stage)
                            
                        if webrtc_ctx.video_processor.feedback_msg:
                            feedback_placeholder.warning(f"{webrtc_ctx.video_processor.feedback_msg}")
                    
                    time.sleep(0.5)

# ==========================================
# 4. App層（起動）
# ==========================================
@st.cache_resource
def get_processor():
    return VideoProcessor()

class App:
    def __init__(self):
        st.set_page_config(page_title="筋トレアプリ", layout="wide")
        SessionManager.init_state()
        self.processor = get_processor()
        self.main_page = MainPageView(self.processor)

    def run(self):
        self.main_page.render()

if __name__ == "__main__":
    app = App()
    app.run()