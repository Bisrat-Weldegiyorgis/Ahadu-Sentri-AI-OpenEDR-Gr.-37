import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

print("Loading data...")
# Load your data - adjust the path as needed
df_train = pd.read_csv("KDDTrain+.txt", header=None)

# Define column names (adjust based on your dataset)
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

df_train.columns = columns[:len(df_train.columns)]

# Prepare features and target
X = df_train.drop('label', axis=1)
y = df_train['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

print("Training model...")
# Fit preprocessor and transform data
X_train_processed = preprocessor.fit_transform(X_train)

# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)  # Small for quick training
model.fit(X_train_processed, y_train)

print("Saving model and preprocessor...")
# Save the newly created files
joblib.dump(model, 'trained_model.pkl')
joblib.dump(preprocessor, 'fitted_preprocessor.pkl')

print(" Model and preprocessor saved successfully!")
print("Files created: 'trained_model.pkl' and 'fitted_preprocessor.pkl'")
