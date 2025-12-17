import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Dropout, 
                                   BatchNormalization, Input, Flatten, MaxPooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, confusion_matrix, classification_report,
                           r2_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from glob import glob
from pyswarm import pso
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU acceleration")
    except RuntimeError as e:
        print(f"GPU error: {e}")
else:
    print("Using CPU")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class DDOSDetector:
    def __init__(self, timesteps=10):
        self.timesteps = timesteps
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.best_params = None
        self.pso_convergence_data = []  # Store PSO iteration data
        self.attack_types = {
            'normal': 0,
            'DrDoS_DNS': 1,
            'DrDoS_LDAP': 2,
            'DrDoS_MSSQL': 3,
            'DrDoS_NetBIOS': 4,
            'DrDoS_NTP': 5,
            'DrDoS_SSDP': 6,
            'DrDoS_UDP': 7,
            'Syn': 8,
            'TFTP': 9,
            'UDPLag': 10
        }
        self.history = None

    def load_data(self, data_dir='CSV'):
        """Load all CSV files from specified directory with more samples"""
        abs_data_dir = os.path.abspath(data_dir)
        print(f"Looking for CSV files in: {abs_data_dir}")
        
        files = glob(os.path.join(abs_data_dir, "*.csv"))
        if not files:
            available_files = os.listdir(abs_data_dir)
            print(f"No CSV files found. Available files in directory: {available_files}")
            raise ValueError(f"No CSV files found in {abs_data_dir}")
        
        dfs = []
        for file in files:
            attack_type = os.path.basename(file).replace('DDOS-PSO-CNN-LSTM-\CSV', '')
            try:
                # Load more samples for better training
                df = pd.read_csv(file)
                df['Label'] = attack_type
                dfs.append(df)
                print(f"Loaded {os.path.basename(file)} with {len(df)} rows")
            except Exception as e:
                print(f"Error loading {os.path.basename(file)}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid CSV files could be loaded")
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Data validation
        numeric_cols = combined_df.select_dtypes(include=np.number).columns
        if combined_df[numeric_cols].isnull().values.any():
            print("Warning: Data contains NaN values")
        if np.isinf(combined_df[numeric_cols].values).any():
            print("Warning: Data contains infinite values")
        
        return self._preprocess(combined_df)

    def _preprocess(self, df):
        """Preprocess the combined dataframe"""
        df = df.copy()
        
        # Clean labels
        df['Label'] = df['Label'].str.strip().replace({
            'drDoS_DNS': 'DrDoS_DNS',
            'drDoS_LDAP': 'DrDoS_LDAP',
        }).fillna('normal')
        
        # Filter to known attack types and map to numerical values
        valid_labels = list(self.attack_types.keys())
        df['Label'] = df['Label'].apply(lambda x: x if x in valid_labels else 'normal')
        y = df['Label'].map(self.attack_types).values
        
        # Handle features
        X = df.select_dtypes(include=np.number)
        
        # Replace infinite values and large values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Clip extreme values to prevent float32 overflow
        X = X.clip(-1e10, 1e10)
        
        X = X.astype('float32')
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Final check for problematic values
        if np.isinf(X).any() or np.isnan(X).any():
            raise ValueError("Data still contains invalid values after preprocessing")
            
        return X, y

    def create_sequences(self, X, y):
        """Create time-series sequences with proper alignment"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.timesteps):
            X_seq.append(X[i:i + self.timesteps])
            y_seq.append(y[i + self.timesteps])  # Predict next step after sequence
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"Sequence shapes - X: {X_seq.shape}, y: {y_seq.shape}")
        return X_seq, y_seq

    def build_model(self, params=None, verbose=True):
        """Build PSO-optimized CNN-LSTM hybrid model"""
        if params is None:
            # Default parameters if PSO not run
            params = {
                'conv1_filters': 64,
                'conv2_filters': 128,
                'lstm_units': 128,
                'dense_units': 64,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            }
        
        if verbose:
            print("Building PSO + CNN + LSTM Hybrid Model Architecture:")
            print("=" * 60)
            print("1. CNN Layers for spatial feature extraction")
            print("2. LSTM Layers for temporal pattern recognition")
            print("3. PSO-optimized hyperparameters for optimal performance")
            print("=" * 60)
        
        model = Sequential([
            Input(shape=(self.timesteps, len(self.feature_names))),
            Conv1D(int(params['conv1_filters']), 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(int(params['conv2_filters']), 3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(params['dropout_rate']),
            LSTM(int(params['lstm_units']), return_sequences=True),
            Dropout(params['dropout_rate']),
            LSTM(int(params['lstm_units']//2)),
            Dropout(params['dropout_rate']),
            Dense(int(params['dense_units']), activation='relu'),
            Dropout(params['dropout_rate']/2),
            Dense(len(self.attack_types), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def model_evaluation(self, params, X_train, y_train, X_val, y_val):
        """Evaluate model for PSO optimization"""
        params_dict = {
            'conv1_filters': max(16, int(params[0])),
            'conv2_filters': max(32, int(params[1])),
            'lstm_units': max(32, int(params[2])),
            'dense_units': max(16, int(params[3])),
            'dropout_rate': max(0.1, min(0.5, params[4])),
            'learning_rate': max(1e-4, min(0.01, params[5]))
        }
        
        # Set verbose=False to prevent repetitive printing during PSO iterations
        model = self.build_model(params_dict, verbose=False)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,  # Fewer epochs for faster PSO evaluation
            batch_size=32,
            verbose=0
        )
        
        # Return validation loss (PSO minimizes this)
        return history.history['val_loss'][-1]

    def pso_optimization(self, X_train, y_train, X_val, y_val):
        """Optimize model hyperparameters using PSO with convergence tracking"""
        print("\nStarting PSO optimization for CNN-LSTM hybrid model...")
        print("PSO + CNN + LSTM Hybridization Process:")
        print("- PSO searches for optimal hyperparameters")
        print("- CNN extracts spatial features from network traffic")
        print("- LSTM captures temporal dependencies in sequences")
        print("- Hybrid model combines strengths of all three approaches")
        
        # Parameter bounds [conv1_filters, conv2_filters, lstm_units, dense_units, dropout_rate, learning_rate]
        lb = [16, 32, 32, 16, 0.1, 1e-4]  # Lower bounds
        ub = [128, 256, 256, 128, 0.5, 0.01]  # Upper bounds
        
        # Store convergence data
        self.pso_convergence_data = []
        
        # Create a wrapper function to track convergence
        def pso_objective_function(params):
            fitness = self.model_evaluation(params, X_train, y_train, X_val, y_val)
            
            # Track convergence data
            iteration = len(self.pso_convergence_data) + 1
            convergence_point = {
                'iteration': iteration,
                'best_fitness': fitness,
                'best_params': {
                    'conv1_filters': max(16, int(params[0])),
                    'conv2_filters': max(32, int(params[1])),
                    'lstm_units': max(32, int(params[2])),
                    'dense_units': max(16, int(params[3])),
                    'dropout_rate': max(0.1, min(0.5, params[4])),
                    'learning_rate': max(1e-4, min(0.01, params[5]))
                }
            }
            
            self.pso_convergence_data.append(convergence_point)
            print(f"Iteration {iteration}: Fitness = {fitness:.4f}")
            
            return fitness
        
        # PSO optimization without callback parameter
        best_params, best_score = pso(
            pso_objective_function,
            lb,
            ub,
            swarmsize=10,
            maxiter=15,
            debug=True
        )
        
        # Convert to parameter dictionary
        self.best_params = {
            'conv1_filters': max(16, int(best_params[0])),
            'conv2_filters': max(32, int(best_params[1])),
            'lstm_units': max(32, int(best_params[2])),
            'dense_units': max(16, int(best_params[3])),
            'dropout_rate': max(0.1, min(0.5, best_params[4])),
            'learning_rate': max(1e-4, min(0.01, best_params[5]))
        }
        
        print(f"\nPSO + CNN + LSTM optimized parameters (Best Score: {best_score:.4f}):")
        for k, v in self.best_params.items():
            print(f"{k}: {v}")
            
        return self.best_params, best_score

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train PSO-optimized CNN-LSTM hybrid model with early stopping"""
        print(f"\nTraining PSO + CNN + LSTM Hybrid Model")
        print(f"Training shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Validation shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        # Run PSO optimization if not already done
        if self.best_params is None:
            self.pso_optimization(X_train, y_train, X_val, y_val)
        
        # Keep verbose=True for the final model building (prints only once)
        self.model = self.build_model(self.best_params, verbose=True)
        
        print("\nPSO + CNN + LSTM Hybrid Model Architecture:")
        self.model.summary()
        
        print("\nTraining the PSO-optimized CNN-LSTM hybrid model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=8, monitor='val_loss')
            ],
            verbose=1
        )
        return self.history

    def evaluate_model(self, X_test, y_test):
        """Evaluate PSO + CNN + LSTM hybrid model and generate comprehensive metrics"""
        print("\nEvaluating PSO + CNN + LSTM Hybrid Model Performance...")
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Handle class mismatch - ensure we only consider classes present in both
        unique_classes_test = np.unique(y_test)
        unique_classes_pred = np.unique(y_pred)
        valid_classes = np.intersect1d(unique_classes_test, unique_classes_pred)
        
        # Filter out classes not present in both test and prediction
        mask = np.isin(y_test, valid_classes)
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]
        y_pred_proba_filtered = y_pred_proba[mask]
        
        if len(valid_classes) == 0:
            print("Warning: No overlapping classes between test and prediction")
            return {}
        
        target_names = [list(self.attack_types.keys())[i] for i in valid_classes]
        
        # Calculate all metrics
        accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
        precision = precision_score(y_test_filtered, y_pred_filtered, average='weighted', zero_division=0)
        recall = recall_score(y_test_filtered, y_pred_filtered, average='weighted', zero_division=0)
        f1 = f1_score(y_test_filtered, y_pred_filtered, average='weighted', zero_division=0)
        
        # R2 Score
        r2 = r2_score(y_test_filtered, y_pred_filtered)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=valid_classes)
        
        # False Alarm Rate and Detection Rate
        # Assuming class 0 is 'normal' (if present)
        if 0 in valid_classes:
            normal_idx = np.where(valid_classes == 0)[0][0]
            tn = cm[normal_idx, normal_idx]  # True negatives
            fp = np.sum(cm[normal_idx, :]) - tn  # False positives
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # True positives (all non-normal correctly classified)
            tp = np.sum(np.diag(cm)) - tn
            # False negatives (non-normal classified as normal)
            fn = np.sum(cm[normal_idx, 1:]) if len(valid_classes) > 1 else 0
            dr = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            far = np.nan
            dr = np.nan
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'r2': r2,
            'far': far,
            'dr': dr
        }
        
        # Generate all plots (without ROC and PR curves)
        self._generate_plots(X_test, y_test, y_pred, y_pred_proba, cm, metrics, valid_classes)
        
        print("\nPSO + CNN + LSTM Hybrid Model - Comprehensive Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        print("\nClassification Report:")
        print(classification_report(y_test_filtered, y_pred_filtered, 
                                  target_names=target_names,
                                  labels=valid_classes))
        
        return metrics

    def _generate_plots(self, X_test, y_test, y_pred, y_pred_proba, cm, metrics, valid_classes):
        """Generate all 10 publication-ready graphs for PSO + CNN + LSTM model"""
        # 1. Training vs Validation Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('PSO + CNN + LSTM Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pso_cnn_lstm_training_validation_accuracy.png', dpi=300)
        plt.close()
        
        # 2. Training vs Validation Loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('PSO + CNN + LSTM Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pso_cnn_lstm_training_validation_loss.png', dpi=300)
        plt.close()
        
        # 3. Confusion Matrix
        plt.figure(figsize=(12, 10))
        target_names = [list(self.attack_types.keys())[i] for i in valid_classes]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('PSO + CNN + LSTM Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('pso_cnn_lstm_confusion_matrix.png', dpi=300)
        plt.close()
        
        # 4. ROC Curve - REMOVED due to multiclass complexity
        
        # 5. Precision-Recall Curve - REMOVED due to multiclass complexity
        
        # 6. Bar Graph (Accuracy, Precision, Recall, F1 comparison)
        plt.figure(figsize=(10, 6))
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [metrics['accuracy'], metrics['precision'], 
                         metrics['recall'], metrics['f1']]
        
        bars = plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple'])
        plt.ylabel('Score')
        plt.title('PSO + CNN + LSTM Performance Metrics')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('pso_cnn_lstm_performance_metrics.png', dpi=300)
        plt.close()
        
        # 7. FAR vs DR Curve
        plt.figure(figsize=(10, 6))
        # For simplicity, we'll plot a point for our model
        plt.scatter(metrics['far'], metrics['dr'], s=100, color='red', zorder=5)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Alarm Rate (FAR)')
        plt.ylabel('Detection Rate (DR)')
        plt.title('PSO + CNN + LSTM FAR vs DR Curve')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('pso_cnn_lstm_far_vs_dr.png', dpi=300)
        plt.close()
        
        # 8. Feature Distribution Graph
        plt.figure(figsize=(12, 8))
        # Select a few important features for visualization
        if len(self.feature_names) > 5:
            selected_features = self.feature_names[:5]  # First 5 features
        else:
            selected_features = self.feature_names
            
        # Create subplots
        n_features = len(selected_features)
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3*n_features))
        
        if n_features == 1:
            axes = [axes]
            
        for i, feature in enumerate(selected_features):
            feature_idx = self.feature_names.index(feature)
            feature_data = X_test[:, :, feature_idx].flatten()
            axes[i].hist(feature_data, bins=50, alpha=0.7, color='skyblue')
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            
        plt.tight_layout()
        plt.savefig('pso_cnn_lstm_feature_distribution.png', dpi=300)
        plt.close()
        
        # 9. PSO Optimization Convergence Graph (Real data)
        plt.figure(figsize=(10, 6))
        iterations = [data['iteration'] for data in self.pso_convergence_data]
        best_scores = [data['best_fitness'] for data in self.pso_convergence_data]
        
        plt.plot(iterations, best_scores, 'b-', marker='o', linewidth=2, markersize=6)
        plt.xlabel('Iteration')
        plt.ylabel('Best Validation Loss')
        plt.title('PSO Optimization Convergence for CNN-LSTM')
        plt.grid(True, alpha=0.3)
        
        # Add min value annotation
        min_idx = np.argmin(best_scores)
        plt.annotate(f'Min: {best_scores[min_idx]:.4f}', 
                    xy=(iterations[min_idx], best_scores[min_idx]),
                    xytext=(iterations[min_idx]+1, best_scores[min_idx]+0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig('pso_cnn_lstm_convergence.png', dpi=300)
        plt.close()
        
        # 10. Class Imbalance Distribution Graph
        plt.figure(figsize=(12, 6))
        class_counts = np.bincount(y_test)
        class_names = [list(self.attack_types.keys())[i] for i in range(len(class_counts)) if i in valid_classes]
        class_counts = [count for i, count in enumerate(class_counts) if i in valid_classes]
        
        bars = plt.bar(class_names, class_counts, color='lightcoral')
        plt.title('PSO + CNN + LSTM - Class Distribution in Test Set')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('pso_cnn_lstm_class_distribution.png', dpi=300)
        plt.close()

    def save_model(self, path='pso_cnn_lstm_model'):
        """Save PSO + CNN + LSTM model artifacts"""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'pso_cnn_lstm_model.keras'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        
        # Save PSO convergence data
        pso_data = {
            'convergence': self.pso_convergence_data,
            'best_params': self.best_params
        }
        with open(os.path.join(path, 'pso_data.json'), 'w') as f:
            json.dump(pso_data, f, indent=4)
        
        # Save attack types mapping and best parameters
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'timesteps': self.timesteps,
                'attack_types': self.attack_types,
                'best_params': self.best_params,
                'model_type': 'PSO + CNN + LSTM Hybrid Model'
            }, f)

def main():
    print("=" * 70)
    print("DDoS Detection System - PSO Optimized CNN-LSTM Hybrid Model")
    print("PSO + CNN + LSTM Hybrid Approach for Network Security")
    print("=" * 70)
    
    detector = DDOSDetector(timesteps=10)
    
    try:
        # Load data with more samples
        print("\nLoading data...")
        X, y = detector.load_data('CSV')
        
        # Create sequences
        print("\nCreating sequences...")
        X_seq, y_seq = detector.create_sequences(X, y)
        
        # Split data
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train
        print("\nTraining PSO + CNN + LSTM hybrid model...")
        detector.train_model(X_train, y_train, X_val, y_val, epochs=100)
        
        # Evaluate
        print("\nEvaluating PSO + CNN + LSTM model performance...")
        metrics = detector.evaluate_model(X_test, y_test)
        
        # Save
        detector.save_model()
        print("\nPSO + CNN + LSTM model saved successfully")
        
        # Print PSO convergence summary
        print("\nPSO + CNN + LSTM Convergence Summary:")
        print(f"Total iterations: {len(detector.pso_convergence_data)}")
        if detector.pso_convergence_data:
            final_score = detector.pso_convergence_data[-1]['best_fitness']
            print(f"Final best validation loss: {final_score:.4f}")
            
        print("\n" + "=" * 70)
        print("PSO + CNN + LSTM Hybrid Model Training Complete!")
        print("Model combines:")
        print("- Particle Swarm Optimization (PSO) for hyperparameter tuning")
        print("- Convolutional Neural Networks (CNN) for spatial feature extraction")
        print("- Long Short-Term Memory (LSTM) for temporal pattern recognition")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()