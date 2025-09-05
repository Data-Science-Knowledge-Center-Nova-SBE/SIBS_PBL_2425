
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def plot_mutual_information(mi_df, output_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=mi_df.head(15), x="MI_Score", y="Feature")
    plt.title("Top 15 Features by Mutual Information")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mutual_information_top15.png"))
    plt.close()

def plot_shap_summary(model, X_test, output_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "shap_summary.png"))
    plt.close()

    # Optional: force plot for one instance (use with caution — not interactive when saved)
    try:
        shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)
        plt.savefig(os.path.join(output_path, "shap_force_plot_0.png"))
        plt.close()
    except Exception:
        pass  # force_plot might fail depending on SHAP version

def plot_predictions(dates, y_true, y_pred, output_path, filename="actual_vs_predicted.png"):
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_true, label='Actual', linewidth=2)
    plt.plot(dates, y_pred, label='Predicted', linewidth=2)
    plt.title("Corrected Actual vs. Predicted Fraud Amount")
    plt.xlabel("Date")
    plt.ylabel("Fraud Amount Accepted")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
