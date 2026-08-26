import os

import streamlit as st
import tempfile

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
        """アップロードされた動画から骨格情報を取得し、動画を生成して返す"""
        if uploaded_file is None:
            return None, None

        # 1. アップロードされた動画を一時ファイルとしてディスクに保存 (OpenCVで読むため)
        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        input_temp.write(uploaded_file.read())
        input_temp.close()

        # 2. 推論を実行して骨格情報(ランドマーク)を取得
        landmarks_data = self.estimator.process_video(input_temp.name)

        # 3. 角度データの取得
        df_angles = self.estimator.get_dataframe()

        counter = SquatCounter()
        frame_counts = [] # 各フレームのカウント数を保存するリスト

        for record in self.estimator.records:
            counter.update_from_pose_angles(record)
            frame_counts.append(counter.count) # そのフレーム時点での回数を保存

        squat_count = counter.count

        # スクワットの評価
        evaluate = SquatEvaluator()
        frame_val = [] 
        fps = 30.0

        for i, landmark in enumerate(landmarks_data):
            current_second = i / fps
            val = evaluate.judge_frame(current_second, landmark)
            frame_val.append(val)
        
        # 最終的な判定結果を取得
        eval_result = evaluate.get_result()

        # 4. 描画済み動画を保存するための一時ファイルを作成
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_temp.close()

        # 5. 取得したランドマークと回数リストを使って、描画済みの動画を生成
        # render_video に frame_counts を渡すように変更
        out_path = self.estimator.render_video(
            input_temp.name, 
            landmarks_data, 
            frame_counts, 
            frame_val,
            output_temp.name
        )


        # ※ 使い終わった入力用の一時ファイルは削除してもOK
        os.remove(input_temp.name)

        return out_path, df_angles, squat_count, eval_result


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
        # 評価結果用のステートを追加
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

        uploaded_video = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "mov", "avi"])

        if uploaded_video is not None:
            st.write("動画")
            st.video(uploaded_video)

            if st.button("解析実行"):
                with st.spinner("解析中"):
                    # 戻り値を4つ受け取る
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

            # 評価サマリーと回数を並べて表示
            col_met1, col_met2 = st.columns(2)
            if processed_count is not None:
                col_met1.metric(label="回数: ", value=f"{processed_count}回")
            if processed_eval is not None:
                col_met2.metric(label="最小膝角度: ", value=f"{processed_eval['min_knee_angle']:.1f}°")
                
                # サマリーメッセージをハイライト表示
                if "GOOD" in processed_eval["summary"]:
                    st.info(processed_eval["summary"])
                else:
                    st.warning(processed_eval["summary"])
                
                # タイムスタンプごとのフィードバックがある場合は折りたたみで表示
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

# ==========================================
# 4. App層（起動）
# ==========================================
@st.cache_resource
def get_processor():
    return VideoProcessor()

class App:
    def __init__(self):
        SessionManager.init_state()
        self.processor = get_processor()
        self.main_page = MainPageView(self.processor)

    def run(self):
        self.main_page.render()

if __name__ == "__main__":
    app = App()
    app.run()