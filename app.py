import os
import tempfile
import time
import cv2
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from pose_estimator import PoseEstimator
from squat_counter import SquatCounter
from judge_squat import SquatEvaluator

# ==========================================
# 1. Logic層（動画の保存や解析処理）
# ==========================================
class VideoProcessor:
    def __init__(self):
        self.estimator = PoseEstimator()

    def process_and_render(self, uploaded_file):
        if uploaded_file is None:
            return None, None

        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        input_temp.write(uploaded_file.read())
        input_temp.close()

        landmarks_data = self.estimator.process_video(input_temp.name)
        df_angles = self.estimator.get_dataframe()

        counter = SquatCounter()
        frame_counts = []

        for record in self.estimator.records:
            counter.update_from_pose_angles(record)
            frame_counts.append(counter.count)

        squat_count = counter.count

        evaluate = SquatEvaluator()
        frame_val = [] 
        fps = 30.0

        for i, landmark in enumerate(landmarks_data):
            current_second = i / fps
            val = evaluate.judge_frame(current_second, landmark)
            frame_val.append(val)
        
        eval_result = evaluate.get_result()

        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_temp.close()

        out_path = self.estimator.render_video(
            input_temp.name, 
            landmarks_data, 
            frame_counts, 
            frame_val,
            output_temp.name
        )

        os.remove(input_temp.name)

        return out_path, df_angles, squat_count, eval_result


# ==========================================
# WebRTC用の映像処理クラス
# ==========================================
class SquatVideoProcessor:
    """WebRTCで取得したフレームを1枚ずつ処理するクラス"""
    def __init__(self):
        # 毎フレーム処理するためのインスタンスを初期化
        self.rt_estimator = PoseEstimator()
        self.rt_evaluator = SquatEvaluator()
        self.rt_counter = SquatCounter()
        self.start_time = time.time()
        
        # UIに渡すためのステータスを保持
        self.current_count = 0
        self.current_stage = "WAITING"
        self.feedback_msg = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # WebRTCから渡されたフレームをOpenCV形式(BGR)に変換
        img = frame.to_ndarray(format="bgr24")
        
        # 鏡のように表示するために左右反転
        img = cv2.flip(img, 1)

        current_time = time.time()
        elapsed_seconds = current_time - self.start_time
        timestamp_ms = int(elapsed_seconds * 1000)

        # 1. 骨格推定と角度計算[cite: 2]
        angles = self.rt_estimator.process_frame(img, timestamp_ms)

        # 2. フォーム判定[cite: 1]
        eval_result = None
        if self.rt_estimator.current_landmarks:
            eval_result = self.rt_evaluator.judge_frame(elapsed_seconds, self.rt_estimator.current_landmarks)
            
            if self.rt_evaluator.feedback_logs:
                latest_log = self.rt_evaluator.feedback_logs[-1]
                if elapsed_seconds - latest_log["time"] < 3.0:
                    self.feedback_msg = latest_log["message"]

        # 3. カウント更新[cite: 3]
        self.current_stage = self.rt_counter.update_from_pose_angles(angles)
        self.current_count = self.rt_counter.count

        # 4. 骨格の描画[cite: 2]
        annotated_frame = self.rt_estimator.draw_landmarks(img.copy())

        # 5. テキストの描画 (動画のrender_videoと同様の処理)[cite: 2]
        l_knee_angle = angles.get('left_hip_left_knee_left_ankle', 0.0) if angles else 0.0
        r_knee_angle = angles.get('right_hip_right_knee_right_ankle', 0.0) if angles else 0.0
        
        cv2.putText(annotated_frame, f"Count: {self.current_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"L-Knee: {l_knee_angle:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"R-Knee: {r_knee_angle:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        if eval_result and "knee_angle" in eval_result:
            eval_angle = eval_result["knee_angle"]
            cv2.putText(annotated_frame, f"Eval Angle: {eval_angle:.1f}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 処理済みの画像をWebRTC側に返す
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
        if "train_count" not in st.session_state:
            st.session_state.train_count = None
        if "eval_result" not in st.session_state:
            st.session_state.eval_result = None

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
            st.write("動画")
            st.video(uploaded_video)

            if st.button("解析実行"):
                with st.spinner("解析中"):
                    out_vid, out_df, out_count, out_eval = self.processor.process_and_render(uploaded_video)
                    
                    SessionManager.set("processed_video_path", out_vid)
                    SessionManager.set("processed_df", out_df)
                    SessionManager.set("train_count", out_count)
                    SessionManager.set("eval_result", out_eval)

        processed_vid = SessionManager.get("processed_video_path")
        processed_df = SessionManager.get("processed_df")
        processed_count = SessionManager.get("train_count")
        processed_eval = SessionManager.get("eval_result")

        if processed_vid and processed_df is not None:
            st.success("解析が完了")

            col_met1, col_met2 = st.columns(2)
            if processed_count is not None:
                col_met1.metric(label="回数: ", value=f"{processed_count}回")
            if processed_eval is not None:
                col_met2.metric(label="最小膝角度: ", value=f"{processed_eval['min_knee_angle']:.1f}°")
                
                if "GOOD" in processed_eval["summary"]:
                    st.info(processed_eval["summary"])
                else:
                    st.warning(processed_eval["summary"])
                
                if processed_eval["feedback_logs"]:
                    with st.expander("フィードバックログの詳細を見る"):
                        for log in processed_eval["feedback_logs"]:
                            st.write(f"- {log['time']:.1f}秒: {log['message']}")
            
            st.divider()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("動画")
                with open(processed_vid, 'rb') as f:
                    st.video(f.read())
            
            with col2:
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
                key="squat-realtime",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=SquatVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        # カメラが動作している間、横にステータスを表示し続けるループ
        with metrics_col:
            st.markdown("### ステータス")
            count_placeholder = st.empty()
            stage_placeholder = st.empty()
            feedback_placeholder = st.empty()
            
            if webrtc_ctx.state.playing:
                while True:
                    if webrtc_ctx.video_processor:
                        # VideoProcessorの属性から現在の値を取得して表示
                        count_placeholder.metric("スクワット回数", f"{webrtc_ctx.video_processor.current_count} 回")
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