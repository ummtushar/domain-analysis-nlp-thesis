import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")

def load_and_preprocess_data(file_path='code_complexity.csv', min_samples_per_class=5):
    """
    Load and preprocess the dataset, filtering out classes with too few samples
    and selecting only the specified feature columns
    
    Args:
        file_path: Path to the CSV file
        min_samples_per_class: Minimum number of samples required for a class to be included
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    unique_domains = df['top1_domain'].unique()
    print(f"Number of unique domains: {len(unique_domains)}")
    
    class_counts = df['top1_domain'].value_counts()
    print("\nTop 10 classes by frequency:")
    print(class_counts.head(10))
    
    valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
    print(f"\nFiltering out classes with fewer than {min_samples_per_class} samples...")
    print(f"Original number of classes: {len(unique_domains)}")
    print(f"Number of classes after filtering: {len(valid_classes)}")
    
    filtered_df = df[df['top1_domain'].isin(valid_classes)]
    print(f"Filtered dataset: {len(filtered_df)} rows (removed {len(df) - len(filtered_df)} rows)")
    
    tensor_lib_features = [col for col in df.columns if col.startswith('tensor_lib_')]
    
    tensor_op_features = [col for col in df.columns if col.startswith('op_')]
    
    specific_features = ['num_lines', 'num_functions', 'num_classes', 
                         'cyclomatic_complexity', 'narrative_size']
    
    feature_cols = specific_features + tensor_lib_features + tensor_op_features
    
    print(f"Using {len(feature_cols)} features:")
    print(f"- Specific features: {specific_features}")
    print(f"- Tensor library features: {tensor_lib_features}")
    print(f"- Tensor operation features: {tensor_op_features}")
    
    X = filtered_df[feature_cols]
    y = filtered_df['top1_domain']
    
    X = X.fillna(0)
    
    return X, y, feature_cols, class_counts, valid_classes, tensor_lib_features, tensor_op_features

def check_multicollinearity(X, feature_names, vif_threshold=10.0, corr_threshold=0.8):
    print("\n=== CHECKING FOR MULTICOLLINEARITY ===")
    
    print("Calculating correlation matrix...")
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr().abs()
    
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]) 
                       for i in range(len(corr_matrix.index)) 
                       for j in range(len(corr_matrix.columns)) 
                       if i < j and abs(corr_matrix.iloc[i, j]) >= corr_threshold]
    
    high_corr_pairs.sort(key=lambda x: -x[2])
    
    print(f"\nFound {len(high_corr_pairs)} feature pairs with correlation >= {corr_threshold}:")
    for feat1, feat2, corr in high_corr_pairs[:10]:  # Show top 10 to keep output manageable
        print(f"- {feat1} & {feat2}: {corr:.4f}")
    if len(high_corr_pairs) > 10:
        print(f"  (and {len(high_corr_pairs) - 10} more pairs)")
    
    to_drop = set()
    print("\nIdentifying features to remove based on correlation...")
    
    processed_pairs = set()
    
    for feat1, feat2, corr in high_corr_pairs:
        pair_key = tuple([feat1, feat2]) # you can sort to drop the highest 
        if pair_key in processed_pairs:
            continue
        
        processed_pairs.add(pair_key)
        
        if feat1 in to_drop and feat2 in to_drop:
            continue
        
        if feat1 in to_drop or feat2 in to_drop:
            continue
        
        if feat1.startswith(('tensor_lib_', 'op_')) and not feat2.startswith(('tensor_lib_', 'op_', 'cyclomatic_complexity')):
            to_drop.add(feat1)
        elif feat2.startswith(('tensor_lib_', 'op_')) and not feat1.startswith(('tensor_lib_', 'op_', 'cyclomatic_complexity')):
            to_drop.add(feat2)
        else:
            to_drop.add(feat2)
    
    print("\nCalculating Variance Inflation Factors...")
    
    remaining_features = [f for f in feature_names if f not in to_drop]
    
    if len(remaining_features) > 1:  # VIF needs at least 2 features
        X_remaining = pd.DataFrame(X, columns=feature_names)[remaining_features]
        
        X_with_const = add_constant(X_remaining)
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_with_const.columns[1:]  # Skip the constant
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                           for i in range(1, X_with_const.shape[1])]
        
        vif_data = vif_data.sort_values("VIF", ascending=False)
        
        print("\nVIF for remaining features (top 10):")
        print(vif_data.head(10))
        
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]["Feature"].tolist()
        
        print(f"\nFeatures with VIF > {vif_threshold}:")
        for feat in high_vif_features:
            print(f"- {feat}: {vif_data[vif_data['Feature'] == feat]['VIF'].values[0]:.4f}")
            to_drop.add(feat)
    
    selected_features = [f for f in feature_names if f not in to_drop]
    
    print(f"\nRemoved {to_drop} feature(s) due to multicollinearity")
    print(f"Kept {len(selected_features)} features")
    
    return selected_features

def calculate_feature_significance(X, y, target_class, feature_names):
    """
    Calculate feature significance using a combination of sklearn's LogisticRegression
    with L1 regularization and permutation-based p-value calculation
    
    Args:
        X: Feature matrix
        y: Target variable
        target_class: The class to model (vs all others)
        feature_names: List of feature names
    
    Returns:
        DataFrame with coefficients, standard errors, p-values, and significance
    """
    
    # Create binary target (1 for target class, 0 for others)
    binary_y = (y == target_class).astype(int)
    
    model = LogisticRegression(
        penalty='l1', 
        C=1.0,  # Regularization strength
        solver='liblinear',  # Works well with L1
        fit_intercept=True,
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X, binary_y)
    
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    
    n_permutations = 1000
    n_samples = X.shape[0]
    n_features = X.shape[1] + 1  # +1 for intercept
    
    permutation_coefs = np.zeros((n_permutations, n_features))
    
    for i in range(n_permutations):
        y_perm = shuffle(binary_y, random_state=i)
        
        try:
            model_perm = LogisticRegression(
                # penalty='l1', 
                # C=1.0,
                # solver='liblinear',
                # fit_intercept=True,
                max_iter=2000,
                random_state=42
            )
            model_perm.fit(X, y_perm)
            
            permutation_coefs[i, 0] = model_perm.intercept_[0]
            permutation_coefs[i, 1:] = model_perm.coef_[0]
        except:
            permutation_coefs[i, :] = 0
            
    std_errors = np.std(permutation_coefs, axis=0)
    
    p_values = np.zeros(n_features)
    for j in range(n_features):
        if coefs[j] >= 0:
            p_values[j] = np.mean(permutation_coefs[:, j] >= coefs[j])
        else:
            p_values[j] = np.mean(permutation_coefs[:, j] <= coefs[j])
    
    lower_ci = coefs - 1.96 * std_errors
    upper_ci = coefs + 1.96 * std_errors
    
    feature_names_with_intercept = ['intercept'] + feature_names
    
    coef_df = pd.DataFrame({
        'Feature': feature_names_with_intercept,
        'Coefficient': coefs,
        'Std.Error': std_errors,
        'Lower_CI': lower_ci,
        'Upper_CI': upper_ci,
        'P_Value': p_values
    })
    
    coef_df['IsSignificant'] = coef_df['P_Value'] < 0.05
    
    return coef_df

# def compute_absolute_importance(coef_df):
#     coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
    
#     # Calculate importance as percentage of total magnitude (excluding intercept)
#     features_only = coef_df[coef_df['Feature'] != 'intercept']
#     total_magnitude = features_only['AbsCoefficient'].sum()
    
#     coef_df['Importance'] = 0.0
#     if total_magnitude > 0:
#         for idx, row in coef_df.iterrows():
#             if row['Feature'] != 'intercept':
#                 coef_df.at[idx, 'Importance'] = (row['AbsCoefficient'] / total_magnitude) * 100
    
#     return coef_df

def train_and_evaluate():
    X, y, feature_cols, class_counts, valid_classes, tensor_lib_features, tensor_op_features = load_and_preprocess_data(min_samples_per_class=5)
    
    selected_feature_names = check_multicollinearity(X, feature_cols, vif_threshold=10.0, corr_threshold=0.8)
    
    X_selected = X[selected_feature_names]
    print(f"\nFinal feature set for modeling: {len(selected_feature_names)} features")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining multinomial logistic regression model...")
    model = LogisticRegression(
        multi_class='multinomial',  
        # solver='saga',             
        # penalty='l1',               
        max_iter=2000,             
        class_weight='balanced',    
        random_state=42,            
        n_jobs=-1,                  # Use all processors
    )
    
    model.fit(X_train_scaled, y_train)
    
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test_scaled)
    
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro-Average F1 Score: {macro_f1:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    all_stats_dfs = []
    
    output_file = 'coefficients_with_significance.txt'
    with open(output_file, 'w') as f:
        f.write("=== MULTINOMIAL LOGISTIC REGRESSION COEFFICIENTS AND SIGNIFICANCE ===\n\n")
        f.write(f"Number of classes: {len(valid_classes)}\n")
        f.write(f"Number of features: {len(selected_feature_names)}\n\n")
        f.write("Note: Multicollinearity was addressed by removing highly correlated features.\n\n")
        f.write("To handle potential singular matrix issues, L1 regularization was applied and\n")
        f.write("p-values were calculated using a permutation approach.\n\n")
        
        for cls in valid_classes:
            try:
                print(f"Calculating coefficients and p-values for class: {cls}...")
                
                n_samples = sum(y_train == cls)
                
                coef_df = calculate_feature_significance(
                    X_train_scaled, y_train, cls, selected_feature_names
                )
                
                if coef_df.empty:
                    print(f"Error calculating coefficients for class {cls}. Skipping.")
                    continue
                
                # coef_df = compute_absolute_importance(coef_df)
                
                coef_df['Class'] = cls
                
                all_stats_dfs.append(coef_df)
                
                f.write(f"\n{'='*80}\n")
                f.write(f"CLASS: {cls}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"Number of training samples: {n_samples}\n\n")
                
                f.write("Coefficients with Significance:\n")
                f.write("-" * 120 + "\n")
                f.write("Feature                      Coefficient    Std.Error     Lower_CI      Upper_CI      P-Value    Significant \n")
                f.write("-" * 120 + "\n")
                
                sorted_coefs = coef_df.sort_values('AbsCoefficient', ascending=False)
                
                for _, row in sorted_coefs.iterrows():
                    feature = row['Feature']
                    coef = row['Coefficient']
                    std_err = row['Std.Error']
                    lower_ci = row['Lower_CI']
                    upper_ci = row['Upper_CI']
                    p_value = row['P_Value']
                    is_sig = "Yes" if row['IsSignificant'] else "No"
                    # importance = row['Importance']
                    
                    if feature == 'intercept':
                        f.write(f"{feature:<30} {coef:>12.6f} {std_err:>12.6f} {lower_ci:>12.6f} {upper_ci:>12.6f} {p_value:>12.6f} {is_sig:>12}      ---\n")
                    else:
                        f.write(f"{feature:<30} {coef:>12.6f} {std_err:>12.6f} {lower_ci:>12.6f} {upper_ci:>12.6f} {p_value:>12.6f} {is_sig:>12} ")
                
                significant_features = coef_df[coef_df['IsSignificant'] & (coef_df['Feature'] != 'intercept')]
                
                if len(significant_features) > 0:
                    f.write("\nSignificant Features Only (p < 0.05):\n")
                    f.write("-" * 100 + "\n")
                    f.write("Feature                      Coefficient    Std.Error     P-Value     Lower_CI      Upper_CI   \n")
                    f.write("-" * 100 + "\n")
                    
                    significant_features = significant_features.sort_values('AbsCoefficient', ascending=False)
                    
                    for _, row in significant_features.iterrows():
                        feature = row['Feature']
                        coef = row['Coefficient']
                        std_err = row['Std.Error']
                        p_value = row['P_Value']
                        lower_ci = row['Lower_CI']
                        upper_ci = row['Upper_CI']
                        # importance = row['Importance']
                        
                        f.write(f"{feature:<30} {coef:>12.6f} {std_err:>12.6f} {p_value:>12.6f} {lower_ci:>12.6f} {upper_ci:>12.6f} \n")
                else:
                    f.write("\nNo significant features found for this class (p < 0.05)\n")
                
            except Exception as e:
                print(f"Error processing class {cls}: {e}")
                
                f.write(f"\n{'='*80}\n")
                f.write(f"CLASS: {cls}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Error during coefficient calculation: {e}\n")
    
    print(f"\nCoefficients with significance written to {output_file}")
    
    if all_stats_dfs:
        combined_stats = pd.concat(all_stats_dfs)
        combined_stats.to_csv('all_coefficient_statistics.csv', index=False)
        print("All coefficient statistics saved to 'all_coefficient_statistics.csv'")
    
    classes = model.classes_
    coef_df = pd.DataFrame(model.coef_, index=classes, columns=selected_feature_names)
    
    intercept_df = pd.DataFrame(model.intercept_, index=classes, columns=['intercept'])
    full_coef_df = pd.concat([intercept_df, coef_df], axis=1)
    
    full_coef_df.to_csv('coefficient_matrix.csv')
    print("Full coefficient matrix saved to 'coefficient_matrix.csv'")
    
    return model, full_coef_df

if __name__ == "__main__":
    train_and_evaluate()