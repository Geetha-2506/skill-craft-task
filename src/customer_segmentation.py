import pandas as pd
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("../data/Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply KMeans
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Save clustered data
df.to_csv("../outputs/task2_customer_segments.csv", index=False)

print("Task-2 completed. Clustered data saved.")
