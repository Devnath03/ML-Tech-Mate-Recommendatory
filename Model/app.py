import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Computer Equipment Recommendation", layout="wide")
st.title("🔍 Computer Equipment Recommendation System")
st.markdown("""
This app suggests computer equipment for your shop based on your selection and current trends. Enter or select an equipment name to get recommendations!
""")

# Load models and data
data = pd.read_pickle("Model/Computer_Equipment_Recommend_Dataset.pickle")
with open("Model/knn_model_equipment.pickle", "rb") as f:
    knn = pickle.load(f)
with open("Model/tfidf_equipment.pickle", "rb") as f:
    vectorizer = pickle.load(f)
with open("Model/scaler_equipment.pickle", "rb") as f:
    scaler = pickle.load(f)

# Feature engineering for input
def get_features(equipment_name):
    row = data[data['Name'].str.lower() == equipment_name.lower()]
    if row.empty:
        return None
    text = row['Name'].values[0] + " " + row['Description'].values[0] + " " + row['Brand'].values[0]
    tfidf = vectorizer.transform([text])
    price = scaler.transform([[row['Price'].values[0]]])
    from scipy.sparse import hstack
    return hstack([tfidf, price])

# Sidebar for input
st.sidebar.header("Input Equipment")
equipment_list = data['Name'].unique().tolist()
equipment_name = st.sidebar.selectbox("Select Equipment Name", equipment_list)
top_n = st.sidebar.slider("Number of Suggestions", min_value=1, max_value=10, value=5)

if st.sidebar.button("Get Recommendations"):
    features = get_features(equipment_name)
    if features is None:
        st.error("Equipment not found.")
    else:
        distances, indices = knn.kneighbors(features, n_neighbors=top_n+1)
        suggestions = data.iloc[indices[0][1:]]  # Exclude the selected item itself
        st.subheader(f"Recommended Equipment for '{equipment_name}':")
        st.dataframe(suggestions[['Name', 'Type', 'Brand', 'Price', 'Description']])

# Show current trends
st.markdown("---")
st.subheader("🔥 Current Trending Equipment")
if 'Trend' in data.columns:
    trending = data[data['Trend'].str.lower() == 'trending']
    st.dataframe(trending[['Name', 'Type', 'Brand', 'Price', 'Description']])
else:
    st.info("No trend data available in the dataset.")

st.markdown("---")
st.caption("Made with Streamlit. Modern UI for computer shop recommendations.")
