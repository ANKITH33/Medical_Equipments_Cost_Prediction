import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_transport_cost_distribution():
    """
    Analyze and visualize the distribution of transport cost
    """

    print("Loading target variable...")
    y_train = pd.read_csv('y_train.csv')

    # If it's a DataFrame, get the series
    if isinstance(y_train, pd.DataFrame):
        if 'Transport_Cost' in y_train.columns:
            y_train = y_train['Transport_Cost']
        else:
            y_train = y_train.iloc[:, 0]  # Take first column

    print(f"Target variable shape: {y_train.shape}")
    print(f"Target variable name: {y_train.name if hasattr(y_train, 'name') else 'Unknown'}")

    # Basic statistics
    print("\n" + "="*60)
    print("TRANSPORT COST DISTRIBUTION ANALYSIS")
    print("="*60)

    print(f"Count: {len(y_train)}")
    print(f"Mean: {y_train.mean():,.2f}")
    print(f"Median: {y_train.median():,.2f}")
    print(f"Std: {y_train.std():,.2f}")
    print(f"Min: {y_train.min():,.2f}")
    print(f"Max: {y_train.max():,.2f}")
    print(f"Skewness: {y_train.skew():.3f}")
    print(f"Kurtosis: {y_train.kurtosis():.3f}")

    # Percentiles
    print(f"\nPercentiles:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(y_train, p)
        print(f"{p:2d}th percentile: {val:12,.2f}")

    # Check for negative values
    negative_count = (y_train < 0).sum()
    print(f"\nNegative values: {negative_count} ({negative_count/len(y_train)*100:.2f}%)")

    # Check for zero values
    zero_count = (y_train == 0).sum()
    print(f"Zero values: {zero_count} ({zero_count/len(y_train)*100:.2f}%)")

    # Outlier analysis using IQR
    Q1 = y_train.quantile(0.25)
    Q3 = y_train.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = ((y_train < lower_bound) | (y_train > upper_bound)).sum()
    print(f"\nOutliers (IQR method): {outliers_iqr} ({outliers_iqr/len(y_train)*100:.2f}%)")
    print(f"IQR bounds: [{lower_bound:,.2f}, {upper_bound:,.2f}]")

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Transport Cost Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Histogram
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Histogram of Transport Cost')
    axes[0, 0].set_xlabel('Transport Cost')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box plot
    axes[0, 1].boxplot(y_train, vert=True)
    axes[0, 1].set_title('Box Plot of Transport Cost')
    axes[0, 1].set_ylabel('Transport Cost')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q-Q plot
    stats.probplot(y_train, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot (Normal Distribution)')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Log-scale histogram (for positive values only)
    positive_values = y_train[y_train > 0]
    if len(positive_values) > 0:
        axes[1, 0].hist(np.log10(positive_values), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Log10 Histogram (Positive Values Only)')
        axes[1, 0].set_xlabel('Log10(Transport Cost)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No positive values for log scale',
                       transform=axes[1, 0].transAxes, ha='center', va='center')
        axes[1, 0].set_title('Log10 Histogram (No positive values)')

    # 5. Density plot
    axes[1, 1].hist(y_train, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title('Density Plot of Transport Cost')
    axes[1, 1].set_xlabel('Transport Cost')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Cumulative distribution
    sorted_values = np.sort(y_train)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    axes[1, 2].plot(sorted_values, cumulative, color='red', linewidth=2)
    axes[1, 2].set_title('Cumulative Distribution')
    axes[1, 2].set_xlabel('Transport Cost')
    axes[1, 2].set_ylabel('Cumulative Probability')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('transport_cost_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Additional detailed analysis for outliers
    print(f"\n" + "="*60)
    print("OUTLIER ANALYSIS FOR DIFFERENT PERCENTILES")
    print("="*60)

    # Analysis for different percentile thresholds
    thresholds = [(2.5, 97.5), (5, 95), (10, 90), (1, 99)]

    for lower_p, upper_p in thresholds:
        lower_threshold = np.percentile(y_train, lower_p)
        upper_threshold = np.percentile(y_train, upper_p)

        outliers = ((y_train < lower_threshold) | (y_train > upper_threshold)).sum()
        remaining = len(y_train) - outliers

        print(f"\nRemoving {lower_p}th and {upper_p}th percentile outliers:")
        print(f"  Thresholds: [{lower_threshold:,.2f}, {upper_threshold:,.2f}]")
        print(f"  Outliers removed: {outliers} ({outliers/len(y_train)*100:.1f}%)")
        print(f"  Remaining samples: {remaining} ({remaining/len(y_train)*100:.1f}%)")

        if remaining > 0:
            filtered_data = y_train[(y_train >= lower_threshold) & (y_train <= upper_threshold)]
            print(f"  Filtered mean: {filtered_data.mean():,.2f}")
            print(f"  Filtered std: {filtered_data.std():,.2f}")
            print(f"  Filtered min: {filtered_data.min():,.2f}")
            print(f"  Filtered max: {filtered_data.max():,.2f}")

    # Create a zoomed-in plot without extreme outliers
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Transport Cost Distribution - Outlier Views', fontsize=14, fontweight='bold')

    # Remove extreme outliers for better visualization
    Q1 = y_train.quantile(0.25)
    Q3 = y_train.quantile(0.75)
    IQR = Q3 - Q1
    filtered_for_plot = y_train[(y_train >= Q1 - 3*IQR) & (y_train <= Q3 + 3*IQR)]

    # Histogram without extreme outliers
    axes[0].hist(filtered_for_plot, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0].set_title(f'Histogram (Removed Extreme Outliers)\n{len(filtered_for_plot)} of {len(y_train)} samples')
    axes[0].set_xlabel('Transport Cost')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    # Box plot comparison
    data_to_plot = [y_train, filtered_for_plot]
    labels = ['All Data', 'Filtered Data']
    axes[1].boxplot(data_to_plot, labels=labels)
    axes[1].set_title('Box Plot Comparison')
    axes[1].set_ylabel('Transport Cost')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('transport_cost_outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Plots saved:")
    print("- transport_cost_distribution.png")
    print("- transport_cost_outlier_analysis.png")

if __name__ == "__main__":
    analyze_transport_cost_distribution()
