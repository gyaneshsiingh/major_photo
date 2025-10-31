# # # pl_spectrum.py
# # import streamlit as st
# # import pandas as pd
# # import os
# # import cv2
# # import numpy as np
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.model_selection import train_test_split
# # import uuid
# # import tempfile
# # from sqlalchemy import create_engine, text
# # import pyvisa
# # import matplotlib.pyplot as plt
# # from datetime import datetime
# # from reportlab.lib.pagesizes import A4
# # from reportlab.pdfgen import canvas
# # from reportlab.lib.units import inch
# # from scipy.interpolate import make_interp_spline

# # # ======================
# # # CONFIG
# # # ======================
# # RESULTS_DB = "sqlite:///pl_results.db"
# # engine = create_engine(RESULTS_DB)
# # ROI = (845, 577, 15, 56)
# # LOCAL_PLOTS_DIR = "saved_plots"
# # os.makedirs(LOCAL_PLOTS_DIR, exist_ok=True)

# # # ======================
# # # MODEL TRAINING
# # # ======================
# # @st.cache_resource
# # def train_model():
# #     df = pd.read_csv("nm RGB.csv")
# #     x = df.drop(columns=["nm"])
# #     y = df["nm"]
# #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# #     model = RandomForestRegressor(n_estimators=150, random_state=42)
# #     model.fit(x_train, y_train)
# #     return model

# # # ======================
# # # FRAME ANALYSIS WITH INTENSITY
# # # ======================
# # def analyze_video_frames(video_path, model):
# #     cap = cv2.VideoCapture(video_path)
# #     if not cap.isOpened():
# #         return {"error": "Unable to open video"}

# #     fps = cap.get(cv2.CAP_PROP_FPS)
# #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     if total_frames == 0 or fps == 0:
# #         return {"error": "Invalid video"}

# #     x, y, w, h = ROI
# #     wavelengths, intensities, times = [], [], []
# #     frame_no = 0

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         roi = frame[y:y+h, x:x+w]
# #         if roi.size == 0:
# #             frame_no += 1
# #             continue

# #         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# #         intensity = np.mean(gray)
# #         avg_color = cv2.resize(roi, (1, 1), interpolation=cv2.INTER_AREA)[0, 0][::-1]
# #         wavelength = model.predict(np.array(avg_color).reshape(1, -1))[0]

# #         wavelengths.append(wavelength)
# #         intensities.append(intensity)
# #         times.append(frame_no / fps)
# #         frame_no += 1

# #     cap.release()
# #     if len(wavelengths) == 0:
# #         return {"error": "No frames processed"}

# #     df = pd.DataFrame({"time_s": times, "wavelength_nm": wavelengths, "intensity": intensities})
# #     stats = {
# #         "avg": np.mean(wavelengths),
# #         "peak": wavelengths[np.argmax(intensities)],
# #         "min": np.min(wavelengths),
# #         "max": np.max(wavelengths),
# #         "duration": times[-1]
# #     }
# #     return {"data": df, "stats": stats}

# # # ======================
# # # DATABASE (AUTO-UPGRADE)
# # # ======================
# # def init_db():
# #     with engine.connect() as conn:
# #         # Base table
# #         conn.execute(text("""
# #             CREATE TABLE IF NOT EXISTS results (
# #                 id TEXT PRIMARY KEY,
# #                 timestamp TEXT,
# #                 sample_name TEXT,
# #                 voltage REAL,
# #                 avg_nm REAL,
# #                 peak_nm REAL,
# #                 min_nm REAL,
# #                 max_nm REAL,
# #                 plot_path TEXT
# #             )
# #         """))
# #         conn.commit()

# #         # Auto-upgrade old tables if needed
# #         existing_cols = pd.read_sql("PRAGMA table_info(results)", conn)['name'].tolist()
# #         expected_cols = ["sample_name", "plot_path", "avg_nm", "peak_nm", "min_nm", "max_nm"]
# #         for col in expected_cols:
# #             if col not in existing_cols:
# #                 if col in ["sample_name", "plot_path"]:
# #                     conn.execute(text(f"ALTER TABLE results ADD COLUMN {col} TEXT"))
# #                 else:
# #                     conn.execute(text(f"ALTER TABLE results ADD COLUMN {col} REAL"))
# #         conn.commit()
# # init_db()

# # def save_summary(entry):
# #     pd.DataFrame([entry]).to_sql("results", engine, if_exists="append", index=False)

# # def load_results():
# #     with engine.connect() as conn:
# #         return pd.read_sql("SELECT * FROM results ORDER BY timestamp DESC", conn)

# # # ======================
# # # PLOTTING
# # # ======================
# # def plot_emission_spectrum(df, sample_name):
# #     if df.empty:
# #         st.info("No spectral data available.")
# #         return

# #     df_sorted = df.sort_values("wavelength_nm")
# #     # Handle duplicate wavelengths by averaging intensities
# #     df_unique = df_sorted.groupby("wavelength_nm", as_index=False)["intensity"].mean()

# #     x = df_unique["wavelength_nm"].values
# #     y = df_unique["intensity"].values

# #     # Smooth if possible
# #     if len(x) > 5:
# #         try:
# #             spline = make_interp_spline(x, y)
# #             x_smooth = np.linspace(min(x), max(x), 400)
# #             y_smooth = spline(x_smooth)
# #         except Exception as e:
# #             st.warning(f"Spline smoothing skipped: {e}")
# #             x_smooth, y_smooth = x, y
# #     else:
# #         x_smooth, y_smooth = x, y

# #     fig, ax = plt.subplots()
# #     ax.plot(x_smooth, y_smooth, color="crimson", linewidth=2)
# #     ax.set_xlabel("Wavelength (nm)")
# #     ax.set_ylabel("Intensity (a.u.)")
# #     ax.set_title(f"Emission Spectrum - {sample_name}")
# #     ax.grid(True)

# #     plot_path = os.path.join(LOCAL_PLOTS_DIR, f"{uuid.uuid4()}.png")
# #     fig.savefig(plot_path, dpi=150, bbox_inches="tight")
# #     st.pyplot(fig)
# #     plt.close(fig)
# #     return plot_path

# # # ======================
# # # MAIN APP
# # # ======================
# # def main():
# #     st.set_page_config(page_title="PL Analyzer", page_icon="⚡", layout="wide")
# #     st.title("⚡ Photoluminescence Spectroscopy Automation System")

# #     model = train_model()
# #     if "voltage" not in st.session_state:
# #         st.session_state.voltage = 0.0

# #     menu = ["📸 Analyze New Sample", "📊 View Previous Results"]
# #     choice = st.sidebar.radio("Menu", menu)

# #     # -------------------------
# #     # ANALYZE NEW SAMPLE
# #     # -------------------------
# #     if choice == "📸 Analyze New Sample":
# #         sample_name = st.text_input("Enter Sample Name 🧪", placeholder="e.g. Si wafer - sample 01")
# #         uploaded = st.file_uploader("Upload Emission Video", type=["mp4", "avi", "mov"])
# #         st.slider("Set Measurement Voltage (V)", 0.0, 5.0, key="voltage")

# #         if uploaded and sample_name:
# #             st.video(uploaded)
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
# #                 tmp.write(uploaded.getbuffer())
# #                 video_path = tmp.name

# #             st.info("⏳ Processing video... extracting spectrum...")
# #             result = analyze_video_frames(video_path, model)
# #             os.remove(video_path)

# #             if "error" in result:
# #                 st.error(result["error"])
# #                 return

# #             df_frames, stats = result["data"], result["stats"]

# #             st.success(f"✅ Processed {len(df_frames)} frames ({stats['duration']:.1f}s)")
# #             st.write(f"**Average λ:** {stats['avg']:.2f} nm")
# #             st.write(f"**Peak λ:** {stats['peak']:.2f} nm")
# #             st.write(f"**Range:** {stats['min']:.2f} – {stats['max']:.2f} nm")

# #             plot_path = plot_emission_spectrum(df_frames, sample_name)

# #             entry = {
# #                 "id": str(uuid.uuid4()),
# #                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# #                 "sample_name": sample_name,
# #                 "voltage": st.session_state.voltage,
# #                 "avg_nm": stats["avg"],
# #                 "peak_nm": stats["peak"],
# #                 "min_nm": stats["min"],
# #                 "max_nm": stats["max"],
# #                 "plot_path": plot_path
# #             }
# #             save_summary(entry)
# #             st.success("Sample results saved successfully ✅")

# #     # -------------------------
# #     # VIEW PREVIOUS RESULTS
# #     # -------------------------
# #     elif choice == "📊 View Previous Results":
# #         df = load_results()

# #         # Drop invalid rows
# #         df = df.dropna(subset=["sample_name"])
# #         if df.empty:
# #             st.warning("No valid sample entries found in database.")
# #             return

# #         sample_options = sorted(df["sample_name"].unique().tolist())
# #         selected = st.selectbox("Select a Sample to View", sample_options)

# #         sample_rows = df[df["sample_name"] == selected]
# #         if sample_rows.empty:
# #             st.warning("No matching data found for this sample.")
# #             return

# #         sample_df = sample_rows.iloc[0]
# #         st.subheader(f"📄 Sample: {sample_df['sample_name']}")
# #         st.write(f"**Recorded on:** {sample_df['timestamp']}")
# #         st.write(f"**Voltage:** {sample_df['voltage']} V")
# #         st.write(f"**Average λ:** {sample_df['avg_nm']:.2f} nm")
# #         st.write(f"**Peak λ:** {sample_df['peak_nm']:.2f} nm")
# #         st.write(f"**Range:** {sample_df['min_nm']:.2f} – {sample_df['max_nm']:.2f} nm")

# #         st.dataframe(sample_rows[["timestamp", "voltage", "avg_nm", "peak_nm", "min_nm", "max_nm"]])

# #         # Show stored plot
# #         latest_plot = sample_df["plot_path"]
# #         if latest_plot and os.path.exists(latest_plot):
# #             st.image(latest_plot, caption="Stored Emission Spectrum")
# #         else:
# #             st.info("Plot not found or not saved.")





# # # ======================
# # if __name__ == "__main__":
# #     main()



## second 2 changes 

# # pl_spectrum.py
# # PL Automation with ensemble modeling (stacking + weighted average)
# import streamlit as st
# import pandas as pd
# import os
# import cv2
# import numpy as np
# import uuid
# import tempfile
# import joblib
# from datetime import datetime
# import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline

# from sqlalchemy import create_engine, text

# # sklearn imports
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score, mean_squared_error

# # Optional XGBoost
# try:
#     from xgboost import XGBRegressor
#     XGB_AVAILABLE = True
# except Exception:
#     XGB_AVAILABLE = False

# # ======================
# # CONFIG
# # ======================
# RESULTS_DB = "sqlite:///pl_results.db"
# engine = create_engine(RESULTS_DB)
# LOCAL_PLOTS_DIR = "saved_plots"
# os.makedirs(LOCAL_PLOTS_DIR, exist_ok=True)
# MODEL_PATH = "best_ensemble_model.joblib"
# WEIGHTED_MODEL_PATH = "best_weighted_model.joblib"

# # ROI fallback size (used only for initial center-based crop if needed)
# DEFAULT_ROI_SIZE = 50

# # ======================
# # MODEL TRAINING -> MULTI MODEL + ENSEMBLE
# # ======================
# @st.cache_resource
# def train_and_select_ensemble(csv_path="nm RGB.csv", force_retrain=False):
#     """
#     Train base models, build two ensemble strategies:
#       - stacking with Ridge meta-learner
#       - weighted-average by validation R2
#     Evaluate on validation set and return the best ensemble (model object) and metadata.
#     Saves the chosen ensemble to disk (MODEL_PATH).
#     """
#     # If saved model exists and not forcing retrain, load it
#     if os.path.exists(MODEL_PATH) and not force_retrain:
#         try:
#             loaded = joblib.load(MODEL_PATH)
#             st.info("Loaded saved ensemble model from disk.")
#             return loaded
#         except Exception:
#             st.warning("Failed to load saved model; retraining...")

#     df = pd.read_csv(csv_path)
#     X = df.drop(columns=["nm"])
#     y = df["nm"].values

#     # split train/validation
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # define base learners
#     base_models = {
#         "rf": RandomForestRegressor(n_estimators=200, random_state=42),
#         "gbr": GradientBoostingRegressor(random_state=42),
#         "svr": SVR(kernel="rbf"),
#         "knn": KNeighborsRegressor(n_neighbors=5)
#     }
#     if XGB_AVAILABLE:
#         base_models["xgb"] = XGBRegressor(n_estimators=200, learning_rate=0.08, random_state=42)

#     # Train base models and collect validation preds
#     val_preds = {}
#     val_scores = {}
#     for name, m in base_models.items():
#         m.fit(X_train, y_train)
#         yp = m.predict(X_val)
#         val_preds[name] = yp
#         r2 = r2_score(y_val, yp)
#         rmse = np.sqrt(mean_squared_error(y_val, yp))
#         val_scores[name] = {"r2": r2, "rmse": rmse}
#         st.write(f"{name}: R2={r2:.4f}, RMSE={rmse:.3f}")

#     # --- Weighted-average ensemble (weights proportional to positive R2) ---
#     # Use max(0, r2) to avoid negative weights; then normalize
#     r2s = np.array([max(0.0, val_scores[name]["r2"]) for name in base_models.keys()])
#     if r2s.sum() == 0:
#         weights = np.ones_like(r2s) / len(r2s)
#     else:
#         weights = r2s / r2s.sum()

#     def weighted_predict(X_input):
#         preds = np.zeros((len(X_input),))
#         for w, (name, model) in zip(weights, base_models.items()):
#             preds += w * model.predict(X_input)
#         return preds

#     # Evaluate weighted ensemble on validation
#     weighted_val_pred = np.zeros_like(y_val, dtype=float)
#     for w, name in zip(weights, base_models.keys()):
#         weighted_val_pred += w * base_models[name].predict(X_val)
#     weighted_r2 = r2_score(y_val, weighted_val_pred)
#     weighted_rmse = np.sqrt(mean_squared_error(y_val, weighted_val_pred))
#     st.write(f"Weighted-avg ensemble: R2={weighted_r2:.4f}, RMSE={weighted_rmse:.3f}")

#     # Save a small wrapper object for weighted model
#     weighted_wrapper = {
#         "type": "weighted",
#         "base_models": base_models,
#         "weights": weights,
#         "val_r2": weighted_r2,
#         "val_rmse": weighted_rmse
#     }

#     # --- Stacking ensemble ---
#     # Prepare estimators list for stacking (tuples)
#     estimators = [(name, model) for name, model in base_models.items()]
#     # Use Ridge as final estimator (fast linear)
#     stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), n_jobs=-1, passthrough=False)
#     stacking.fit(X_train, y_train)
#     st_pred = stacking.predict(X_val)
#     stacking_r2 = r2_score(y_val, st_pred)
#     stacking_rmse = np.sqrt(mean_squared_error(y_val, st_pred))
#     st.write(f"Stacking ensemble: R2={stacking_r2:.4f}, RMSE={stacking_rmse:.3f}")

#     # Decide best ensemble (compare stacking vs weighted)
#     if stacking_r2 >= weighted_r2:
#         chosen = {
#             "type": "stacking",
#             "model": stacking,
#             "val_r2": stacking_r2,
#             "val_rmse": stacking_rmse,
#             "base_models": base_models
#         }
#         joblib.dump(chosen, MODEL_PATH)
#         st.success(f"Selected Stacking ensemble (R2={stacking_r2:.4f})")
#     else:
#         chosen = weighted_wrapper
#         joblib.dump(chosen, MODEL_PATH)
#         st.success(f"Selected Weighted-average ensemble (R2={weighted_r2:.4f})")

#     return chosen

# # ======================
# # MODEL PREDICTION WRAPPER
# # ======================
# def ensemble_predict(ensemble_obj, X_input):
#     """
#     X_input: numpy array shape (n_samples, n_features)
#     ensemble_obj: the dict saved by train_and_select_ensemble
#     """
#     if ensemble_obj is None:
#         raise RuntimeError("No ensemble model provided")

#     if ensemble_obj.get("type") == "stacking":
#         model = ensemble_obj["model"]
#         return model.predict(X_input)
#     elif ensemble_obj.get("type") == "weighted":
#         preds = np.zeros((len(X_input),))
#         for w, (name, m) in zip(ensemble_obj["weights"], ensemble_obj["base_models"].items()):
#             preds += w * m.predict(X_input)
#         return preds
#     else:
#         raise RuntimeError("Unknown ensemble type")

# # ======================
# # DYNAMIC ROI DETECTION
# # ======================
# def detect_dynamic_roi(frame, roi_size=DEFAULT_ROI_SIZE):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (9, 9), 0)
#     _, _, _, maxLoc = cv2.minMaxLoc(gray)
#     (xc, yc) = maxLoc
#     x1, y1 = max(0, xc - roi_size//2), max(0, yc - roi_size//2)
#     x2, y2 = min(frame.shape[1], xc + roi_size//2), min(frame.shape[0], yc + roi_size//2)
#     roi = frame[y1:y2, x1:x2]
#     return roi, (x1, y1, x2, y2)

# # ======================
# # FRAME ANALYSIS (per-frame wavelength & intensity)
# # ======================
# def analyze_video_frames(video_path, ensemble_obj):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return {"error": "Unable to open video"}

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total_frames == 0 or fps == 0:
#         return {"error": "Invalid video (fps/frames)"}

#     wavelengths, intensities, times = [], [], []
#     frame_no = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         roi, coords = detect_dynamic_roi(frame, roi_size=DEFAULT_ROI_SIZE)
#         if roi.size == 0:
#             frame_no += 1
#             continue

#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         intensity = float(np.mean(gray))

#         # get average RGB and predict with ensemble
#         avg_color_bgr = cv2.resize(roi, (1, 1), interpolation=cv2.INTER_AREA)[0, 0]
#         avg_color_rgb = avg_color_bgr[::-1].astype(float)
#         X_in = np.array(avg_color_rgb).reshape(1, -1)
#         nm_pred = float(ensemble_predict(ensemble_obj, X_in)[0])

#         wavelengths.append(nm_pred)
#         intensities.append(intensity)
#         times.append(frame_no / fps)
#         frame_no += 1

#     cap.release()
#     if len(wavelengths) == 0:
#         return {"error": "No frames processed"}

#     df = pd.DataFrame({"time_s": times, "wavelength_nm": wavelengths, "intensity": intensities})
#     stats = {
#         "avg": float(np.mean(wavelengths)),
#         "peak": float(np.max(wavelengths)),  # or peak by intensity using intensities argmax if you want
#         "min": float(np.min(wavelengths)),
#         "max": float(np.max(wavelengths)),
#         "duration": float(times[-1])
#     }
#     return {"data": df, "stats": stats}

# # ======================
# # DB and plotting helpers
# # ======================
# def init_db():
#     with engine.connect() as conn:
#         conn.execute(text("""
#             CREATE TABLE IF NOT EXISTS results (
#                 id TEXT PRIMARY KEY,
#                 timestamp TEXT,
#                 sample_name TEXT,
#                 voltage REAL,
#                 avg_nm REAL,
#                 peak_nm REAL,
#                 min_nm REAL,
#                 max_nm REAL,
#                 plot_path TEXT,
#                 ensemble_type TEXT,
#                 ensemble_r2 REAL
#             )
#         """))
#         conn.commit()
# init_db()

# def save_summary(entry):
#     pd.DataFrame([entry]).to_sql("results", engine, if_exists="append", index=False)

# def load_results():
#     with engine.connect() as conn:
#         return pd.read_sql("SELECT * FROM results ORDER BY timestamp DESC", conn)

# def plot_emission_spectrum(df, sample_name):
#     df_sorted = df.sort_values("wavelength_nm")
#     df_unique = df_sorted.groupby("wavelength_nm", as_index=False)["intensity"].mean()
#     x = df_unique["wavelength_nm"].values
#     y = df_unique["intensity"].values

#     if len(x) > 5:
#         try:
#             spline = make_interp_spline(x, y)
#             x_s = np.linspace(x.min(), x.max(), 400)
#             y_s = spline(x_s)
#         except Exception as e:
#             st.warning(f"Spline skipped: {e}")
#             x_s, y_s = x, y
#     else:
#         x_s, y_s = x, y

#     fig, ax = plt.subplots()
#     ax.plot(x_s, y_s, color="crimson", lw=2)
#     ax.set_xlabel("Wavelength (nm)")
#     ax.set_ylabel("Intensity (a.u.)")
#     ax.set_title(f"Emission Spectrum — {sample_name}")
#     ax.grid(True)
#     plot_path = os.path.join(LOCAL_PLOTS_DIR, f"{uuid.uuid4()}.png")
#     fig.savefig(plot_path, dpi=150, bbox_inches="tight")
#     st.pyplot(fig)
#     plt.close(fig)
#     return plot_path

# # ======================
# # STREAMLIT APP
# # ======================
# def main():
#     st.set_page_config(page_title="PL Ensemble System", layout="wide")
#     st.title("Photoluminescence Analyzer — Ensemble Mode")

#     # Train/select ensemble (or load saved)
#     ensemble_obj = train_and_select_ensemble()

#     # get ensemble metadata if available
#     ensemble_type = ensemble_obj.get("type", "stacking")
#     ensemble_r2 = float(ensemble_obj.get("val_r2", np.nan)) if isinstance(ensemble_obj, dict) else np.nan

#     st.sidebar.markdown("### Model")
#     st.sidebar.write(f"Ensemble: **{ensemble_type}** (val R²={ensemble_r2:.4f})")
#     if st.sidebar.button("🔁 Retrain models"):
#         # force retrain by removing saved model and calling train
#         if os.path.exists(MODEL_PATH):
#             os.remove(MODEL_PATH)
#         ensemble_obj = train_and_select_ensemble(force_retrain=True)

#     menu = ["Analyze New Sample", "View Results"]
#     choice = st.sidebar.radio("Menu", menu)

#     if choice == "Analyze New Sample":
#         sample_name = st.text_input("Sample name", "")
#         uploaded = st.file_uploader("Upload emission video", type=["mp4", "avi", "mov"])
#         voltage = st.slider("Voltage (V)", 0.0, 10.0, 0.0)

#         if uploaded and sample_name.strip() != "":
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#                 tmp.write(uploaded.getbuffer())
#                 vid_path = tmp.name

#             st.info("Processing — this may take a while...")
#             res = analyze_video_frames(vid_path, ensemble_obj)
#             os.remove(vid_path)

#             if "error" in res:
#                 st.error(res["error"])
#             else:
#                 df_frames = res["data"]
#                 stats = res["stats"]
#                 st.write(f"Duration: {stats['duration']:.2f}s — Frames: {len(df_frames)}")
#                 st.write(f"Average λ: {stats['avg']:.2f} nm")
#                 st.write(f"Min/Max λ: {stats['min']:.2f} — {stats['max']:.2f} nm")
#                 st.write(f"Peak λ: {stats['peak']:.2f} nm")

#                 plot_path = plot_emission_spectrum(df_frames, sample_name)

#                 entry = {
#                     "id": str(uuid.uuid4()),
#                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "sample_name": sample_name,
#                     "voltage": float(voltage),
#                     "avg_nm": float(stats["avg"]),
#                     "peak_nm": float(stats["peak"]),
#                     "min_nm": float(stats["min"]),
#                     "max_nm": float(stats["max"]),
#                     "plot_path": plot_path,
#                     "ensemble_type": ensemble_type,
#                     "ensemble_r2": float(ensemble_r2)
#                 }
#                 save_summary(entry)
#                 st.success("Saved run and plot.")

#     elif choice == "View Results":
#         df = load_results()
#         if df.empty:
#             st.info("No results yet.")
#             return
#         df = df.dropna(subset=["sample_name"])
#         sample_list = sorted(df["sample_name"].unique().tolist())
#         sel = st.selectbox("Select sample", sample_list)
#         rows = df[df["sample_name"] == sel]
#         if rows.empty:
#             st.warning("No rows found for selection.")
#             return
#         # show all runs for this sample
#         st.dataframe(rows[["timestamp", "voltage", "avg_nm", "peak_nm", "min_nm", "max_nm", "ensemble_type"]])
#         latest = rows.iloc[0]
#         st.write("Latest run:")
#         st.write(f"Timestamp: {latest['timestamp']}")
#         st.write(f"Voltage: {latest['voltage']}")
#         st.write(f"Avg λ: {latest['avg_nm']:.2f} nm — Peak: {latest['peak_nm']:.2f} nm — Range: {latest['min_nm']:.2f}-{latest['max_nm']:.2f} nm")
#         if latest["plot_path"] and os.path.exists(latest["plot_path"]):
#             st.image(latest["plot_path"], caption="Stored emission spectrum")
#         else:
#             st.info("Plot file not found.")

# if __name__ == "__main__":
#     main()


#3

# import streamlit as st
# st.set_page_config(page_title="PL Ensemble System", layout="wide")

# import pandas as pd
# import os
# import cv2
# import numpy as np
# import uuid
# import tempfile
# import joblib
# from datetime import datetime
# import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline
# from sqlalchemy import create_engine, text

# # sklearn imports
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score, mean_squared_error

# # Optional XGBoost
# try:
#     from xgboost import XGBRegressor
#     XG_AVAILABLE = True
# except Exception:
#     XG_AVAILABLE = False

# # ======================
# # CONFIG
# # ======================
# RESULTS_DB = "sqlite:///pl_results.db"
# engine = create_engine(RESULTS_DB)
# LOCAL_PLOTS_DIR = "saved_plots"
# os.makedirs(LOCAL_PLOTS_DIR, exist_ok=True)
# MODEL_PATH = "best_ensemble_model.joblib"
# DEFAULT_ROI_SIZE = 50

# # ======================
# # AUTO DB INIT / FIXER
# # ======================
# def init_db():
#     with engine.connect() as conn:
#         conn.execute(text("""
#             CREATE TABLE IF NOT EXISTS results (
#                 id TEXT PRIMARY KEY,
#                 timestamp TEXT,
#                 sample_name TEXT,
#                 voltage REAL,
#                 avg_nm REAL,
#                 peak_nm REAL,
#                 min_nm REAL,
#                 max_nm REAL,
#                 plot_path TEXT,
#                 ensemble_type TEXT,
#                 ensemble_r2 REAL
#             )
#         """))
#         conn.commit()

# def ensure_db_columns():
#     """Ensures all columns exist even if DB is old."""
#     with engine.connect() as conn:
#         existing_cols = pd.read_sql("PRAGMA table_info(results);", conn)["name"].tolist()
#         missing = []
#         if "ensemble_type" not in existing_cols:
#             conn.execute(text("ALTER TABLE results ADD COLUMN ensemble_type TEXT"))
#             missing.append("ensemble_type")
#         if "ensemble_r2" not in existing_cols:
#             conn.execute(text("ALTER TABLE results ADD COLUMN ensemble_r2 REAL"))
#             missing.append("ensemble_r2")
#         if missing:
#             st.warning(f"🔧 Added columns to DB: {', '.join(missing)}")
#         conn.commit()

# init_db()
# ensure_db_columns()

# # ======================
# # ENSEMBLE TRAINING
# # ======================
# @st.cache_resource
# def train_and_select_ensemble(csv_path="nm RGB.csv", force_retrain=False):
#     """Train base models, build ensembles, and select best."""
#     if os.path.exists(MODEL_PATH) and not force_retrain:
#         try:
#             loaded = joblib.load(MODEL_PATH)
#             st.info("Loaded saved ensemble model.")
#             return loaded
#         except Exception:
#             st.warning("Failed to load model, retraining...")

#     df = pd.read_csv(csv_path)
#     X = df.drop(columns=["nm"])
#     y = df["nm"].values
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     base_models = {
#         "rf": RandomForestRegressor(n_estimators=200, random_state=42),
#         "gbr": GradientBoostingRegressor(random_state=42),
#         "svr": SVR(kernel="rbf"),
#         "knn": KNeighborsRegressor(n_neighbors=5)
#     }
#     if XG_AVAILABLE:
#         base_models["xgb"] = XGBRegressor(n_estimators=200, learning_rate=0.08, random_state=42)

#     val_scores = {}
#     for name, model in base_models.items():
#         model.fit(X_train, y_train)
#         pred = model.predict(X_val)
#         val_scores[name] = r2_score(y_val, pred)
#         st.write(f"{name} → R² = {val_scores[name]:.4f}")

#     # Weighted ensemble
#     r2_vals = np.array([max(0, s) for s in val_scores.values()])
#     weights = r2_vals / (r2_vals.sum() if r2_vals.sum() else 1)

#     def weighted_predict(X_input):
#         return sum(w * m.predict(X_input) for w, m in zip(weights, base_models.values()))

#     weighted_pred = weighted_predict(X_val)
#     weighted_r2 = r2_score(y_val, weighted_pred)
#     st.write(f"Weighted Ensemble R² = {weighted_r2:.4f}")

#     # Stacking ensemble
#     estimators = [(n, m) for n, m in base_models.items()]
#     stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), n_jobs=-1)
#     stack.fit(X_train, y_train)
#     stack_pred = stack.predict(X_val)
#     stack_r2 = r2_score(y_val, stack_pred)
#     st.write(f"Stacking Ensemble R² = {stack_r2:.4f}")

#     # Choose best
#     if stack_r2 >= weighted_r2:
#         chosen = {"type": "stacking", "model": stack, "r2": stack_r2}
#         joblib.dump(chosen, MODEL_PATH)
#         st.success(f"✅ Selected Stacking Ensemble (R²={stack_r2:.3f})")
#     else:
#         chosen = {"type": "weighted", "models": base_models, "weights": weights, "r2": weighted_r2}
#         joblib.dump(chosen, MODEL_PATH)
#         st.success(f"✅ Selected Weighted Ensemble (R²={weighted_r2:.3f})")
#     return chosen

# # ======================
# # ENSEMBLE PREDICTION
# # ======================
# def ensemble_predict(model_obj, X_input):
#     if model_obj["type"] == "stacking":
#         return model_obj["model"].predict(X_input)
#     elif model_obj["type"] == "weighted":
#         preds = np.zeros(len(X_input))
#         for w, m in zip(model_obj["weights"], model_obj["models"].values()):
#             preds += w * m.predict(X_input)
#         return preds
#     else:
#         raise ValueError("Invalid ensemble object")

# # ======================
# # ROI DETECTION
# # ======================
# def detect_dynamic_roi(frame, roi_size=DEFAULT_ROI_SIZE):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (9, 9), 0)
#     _, _, _, maxLoc = cv2.minMaxLoc(gray)
#     xc, yc = maxLoc
#     x1, y1 = max(0, xc - roi_size//2), max(0, yc - roi_size//2)
#     x2, y2 = min(frame.shape[1], xc + roi_size//2), min(frame.shape[0], yc + roi_size//2)
#     roi = frame[y1:y2, x1:x2]
#     return roi

# # ======================
# # VIDEO FRAME ANALYSIS
# # ======================
# def analyze_video_frames(video_path, ensemble_obj):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return {"error": "Cannot open video"}
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if fps == 0 or total_frames == 0:
#         return {"error": "Invalid video file"}

#     wavelengths, intensities = [], []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         roi = detect_dynamic_roi(frame)
#         if roi.size == 0:
#             continue
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         intensity = float(np.mean(gray))
#         avg_rgb = cv2.resize(roi, (1, 1))[0, 0][::-1]
#         pred_nm = ensemble_predict(ensemble_obj, np.array(avg_rgb).reshape(1, -1))[0]
#         wavelengths.append(pred_nm)
#         intensities.append(intensity)
#     cap.release()

#     if not wavelengths:
#         return {"error": "No frames processed"}

#     df = pd.DataFrame({"wavelength_nm": wavelengths, "intensity": intensities})
#     stats = {
#         "avg": float(np.mean(wavelengths)),
#         "peak": float(np.max(wavelengths)),
#         "min": float(np.min(wavelengths)),
#         "max": float(np.max(wavelengths))
#     }
#     return {"data": df, "stats": stats}

# # ======================
# # EMISSION SPECTRUM PLOT (WAVELENGTH vs INTENSITY)
# # ======================
# def plot_emission_spectrum(df, sample_name):
#     df_sorted = df.sort_values("wavelength_nm")
#     df_unique = df_sorted.groupby("wavelength_nm", as_index=False)["intensity"].mean()
#     x = df_unique["wavelength_nm"].values
#     y = df_unique["intensity"].values
#     y = y / np.max(y)
#     if len(x) > 5:
#         try:
#             spline = make_interp_spline(x, y)
#             x_s = np.linspace(x.min(), x.max(), 400)
#             y_s = spline(x_s)
#         except Exception:
#             x_s, y_s = x, y
#     else:
#         x_s, y_s = x, y
#     peak_idx = np.argmax(y_s)
#     peak_wavelength = x_s[peak_idx]
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(x_s, y_s, color="royalblue", lw=2)
#     ax.scatter(peak_wavelength, y_s[peak_idx], color="red", s=50, label=f"Peak: {peak_wavelength:.1f} nm")
#     ax.set_xlabel("Wavelength (nm)")
#     ax.set_ylabel("Normalized Intensity (a.u.)")
#     ax.set_title(f"Emission Spectrum — {sample_name}")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     path = os.path.join(LOCAL_PLOTS_DIR, f"{uuid.uuid4()}.png")
#     fig.savefig(path, dpi=150, bbox_inches="tight")
#     st.pyplot(fig)
#     plt.close(fig)
#     return path, peak_wavelength

# # ======================
# # MAIN APP
# # ======================
# def main():
#     st.title("⚡ Photoluminescence Spectroscopy — Ensemble Model")

#     ensemble_obj = train_and_select_ensemble()
#     ensemble_type = ensemble_obj.get("type", "stacking")
#     ensemble_r2 = float(ensemble_obj.get("r2", np.nan))

#     st.sidebar.write(f"Model: **{ensemble_type}** | R²={ensemble_r2:.3f}")
#     if st.sidebar.button("🔁 Retrain Models"):
#         if os.path.exists(MODEL_PATH):
#             os.remove(MODEL_PATH)
#         ensemble_obj = train_and_select_ensemble(force_retrain=True)

#     menu = ["Analyze New Sample", "View Results"]
#     choice = st.sidebar.radio("Menu", menu)

#     if choice == "Analyze New Sample":
#         sample_name = st.text_input("Sample Name")
#         uploaded = st.file_uploader("Upload Emission Video", type=["mp4", "avi", "mov"])
#         voltage = st.slider("Voltage (V)", 0.0, 10.0, 0.0)
#         if uploaded and sample_name.strip():
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#                 tmp.write(uploaded.getbuffer())
#                 path = tmp.name
#             st.info("Processing video...")
#             result = analyze_video_frames(path, ensemble_obj)
#             os.remove(path)
#             if "error" in result:
#                 st.error(result["error"])
#             else:
#                 df = result["data"]
#                 stats = result["stats"]
#                 st.write(f"Average λ: {stats['avg']:.2f} nm | Range: {stats['min']:.2f}-{stats['max']:.2f} nm")
#                 st.write(f"Peak λ: {stats['peak']:.2f} nm")
#                 plot_path, peak_wl = plot_emission_spectrum(df, sample_name)
#                 entry = {
#                     "id": str(uuid.uuid4()),
#                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "sample_name": sample_name,
#                     "voltage": float(voltage),
#                     "avg_nm": float(stats["avg"]),
#                     "peak_nm": float(peak_wl),
#                     "min_nm": float(stats["min"]),
#                     "max_nm": float(stats["max"]),
#                     "plot_path": plot_path,
#                     "ensemble_type": ensemble_type,
#                     "ensemble_r2": float(ensemble_r2)
#                 }
#                 pd.DataFrame([entry]).to_sql("results", engine, if_exists="append", index=False)
#                 st.success("Saved successfully ✅")

#     elif choice == "View Results":
#         df = pd.read_sql("SELECT * FROM results", engine)
#         if df.empty:
#             st.info("No results found yet.")
#             return
#         for col in ["ensemble_type", "ensemble_r2"]:
#             if col not in df.columns:
#                 df[col] = "N/A"
#         samples = sorted(df["sample_name"].dropna().unique().tolist())
#         selected = st.selectbox("Select Sample", samples)
#         rows = df[df["sample_name"] == selected]
#         display_cols = [c for c in ["timestamp", "voltage", "avg_nm", "peak_nm", "min_nm", "max_nm", "ensemble_type"] if c in rows.columns]
#         st.dataframe(rows[display_cols])
#         if "plot_path" in rows.columns and os.path.exists(rows.iloc[0]["plot_path"]):
#             st.image(rows.iloc[0]["plot_path"], caption=f"Spectrum — {selected}")

# if __name__ == "__main__":
#     main()