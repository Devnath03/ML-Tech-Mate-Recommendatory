# Computer Shop Equipment Recommender

A machine learning-powered recommendation system for computer shop equipment. Suggests related products and current trends using an interactive Streamlit web app.

## Features
- Unified recommendations for GPUs, CPUs, and other components
- Interactive Streamlit frontend for easy use
- Trending equipment highlights
- Extensible for new product categories

## Demo
<img width="1886" height="881" alt="image" src="https://github.com/user-attachments/assets/eeac18a2-87b8-4290-8059-bc44035214c5" />

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- Recommended: virtualenv

### Installation
```bash
# Clone the repository
git clone https://github.com/Devnath03/ML-Tech-Mate-Recommendatory.git
cd ML-Tech-Mate-Recommendatory/Model

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
streamlit run app.py
```

## Project Structure
```
Model/
├── app.py                      # Streamlit frontend application
├── Computer_Equipment_Recommendation.ipynb  # Model training notebook
├── Computer_Equipment_Recommend_Dataset.pickle  # Processed dataset
├── knn_model_equipment.pickle  # Trained KNN model
├── scaler_equipment.pickle     # Scaler for price features
├── tfidf_equipment.pickle      # TF-IDF vectorizer
├── datasets/                   # Raw CSV datasets
│   ├── All_GPUs.csv
│   ├── CPU.csv
│   └── datasets.csv
└── README.md                   # Project documentation
```

## How It Works
1. **Data Preparation:** Combines multiple equipment datasets (GPUs, CPUs, etc.)
2. **Feature Engineering:** Extracts text and numerical features for recommendations
3. **Model Training:** Uses KNN and TF-IDF for similarity-based suggestions
4. **Frontend:** Streamlit app for user interaction and recommendations

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Author
- [Devnath03](https://github.com/Devnath03)

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
