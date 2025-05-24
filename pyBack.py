#!/usr/bin/env python3
"""
Advanced Mass Balance Reconciliation System
Provides enhanced algorithms and data processing capabilities
"""

import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedMassBalanceReconciler:
    """
    Advanced mass balance reconciliation using weighted least squares
    with constraint optimization and statistical validation
    """
    
    def __init__(self):
        self.streams = []
        self.constraints = []
        self.reconciled_data = None
        self.chi_square_test = None
        self.db_path = 'mass_balance_history.db'
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database for historical data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reconciliation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                original_imbalance REAL,
                final_imbalance REAL,
                chi_square_statistic REAL,
                degrees_of_freedom INTEGER,
                p_value REAL,
                efficiency REAL,
                streams_data TEXT,
                reconciled_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stream_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                measured_value REAL,
                reconciled_value REAL,
                uncertainty REAL,
                adjustment REAL,
                normalized_residual REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_streams(self, streams_data):
        """Load stream data from JSON format"""
        self.streams = streams_data
        return self
    
    def add_constraint(self, constraint_type, streams, target_value=0):
        """
        Add mass balance constraints
        constraint_type: 'mass_balance', 'ratio', 'equality'
        streams: list of stream indices or names
        """
        self.constraints.append({
            'type': constraint_type,
            'streams': streams,
            'target': target_value
        })
        return self
    
    def weighted_least_squares_reconciliation(self):
        """
        Perform advanced weighted least squares reconciliation
        with constraint optimization
        """
        n_streams = len(self.streams)
        
        # Extract measured values and uncertainties
        measured = np.array([s['measured'] for s in self.streams])
        uncertainties = np.array([s['uncertainty'] for s in self.streams])
        
        # Weight matrix (inverse of covariance matrix)
        W = np.diag(1.0 / (uncertainties ** 2))
        
        # Build constraint matrix A and constraint vector b
        A, b = self._build_constraint_matrix()
        
        # Objective function: minimize weighted sum of squared residuals
        def objective(x):
            residuals = x - measured
            return residuals.T @ W @ residuals
        
        # Constraint function for equality constraints
        def constraint_func(x):
            return A @ x - b
        
        # Optimization constraints
        constraints = {'type': 'eq', 'fun': constraint_func}
        
        # Initial guess (measured values)
        x0 = measured.copy()
        
        # Bounds (non-negative values for most streams)
        bounds = [(0, None) for _ in range(n_streams)]
        
        # Solve optimization problem
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            reconciled = measured.copy()
        else:
            reconciled = result.x
        
        # Calculate statistics
        residuals = reconciled - measured
        normalized_residuals = residuals / uncertainties
        
        # Chi-square test for data consistency
        chi_square_stat = np.sum(normalized_residuals ** 2)
        degrees_of_freedom = len(measured) - len(b)  # n_measurements - n_constraints
        p_value = 1 - chi2.cdf(chi_square_stat, degrees_of_freedom)
        
        # Store results
        self.reconciled_data = {
            'reconciled_values': reconciled.tolist(),
            'residuals': residuals.tolist(),
            'normalized_residuals': normalized_residuals.tolist(),
            'chi_square_statistic': chi_square_stat,
            'degrees_of_freedom': degrees_of_freedom,
            'p_value': p_value,
            'data_consistency': 'PASS' if p_value > 0.05 else 'FAIL'
        }
        
        # Update stream data with reconciled values
        for i, stream in enumerate(self.streams):
            stream['reconciled'] = float(reconciled[i])
            stream['adjustment'] = float(residuals[i])
            stream['normalized_residual'] = float(normalized_residuals[i])
            
            # Status based on normalized residuals
            abs_norm_res = abs(normalized_residuals[i])
            if abs_norm_res < 1.0:
                stream['status'] = 'good'
            elif abs_norm_res < 2.0:
                stream['status'] = 'warning'
            else:
                stream['status'] = 'error'
        
        # Save to database
        self._save_to_database()
        
        return self
    
    def _build_constraint_matrix(self):
        """Build constraint matrix for mass balance"""
        n_streams = len(self.streams)
        
        # Default: overall mass balance constraint
        A = np.zeros((1, n_streams))
        b = np.zeros(1)
        
        for i, stream in enumerate(self.streams):
            if stream['type'] == 'input':
                A[0, i] = 1
            else:  # output
                A[0, i] = -1
        
        # Add custom constraints
        for constraint in self.constraints:
            if constraint['type'] == 'mass_balance':
                # Additional mass balance constraints for sub-systems
                new_row = np.zeros(n_streams)
                for stream_idx in constraint['streams']:
                    new_row[stream_idx] = 1 if self.streams[stream_idx]['type'] == 'input' else -1
                A = np.vstack([A, new_row])
                b = np.append(b, constraint['target'])
        
        return A, b
    
    def _save_to_database(self):
        """Save reconciliation results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Calculate metrics
        inputs = [s for s in self.streams if s['type'] == 'input']
        outputs = [s for s in self.streams if s['type'] == 'output']
        
        original_imbalance = sum(s['measured'] for s in inputs) - sum(s['measured'] for s in outputs)
        final_imbalance = sum(s['reconciled'] for s in inputs) - sum(s['reconciled'] for s in outputs)
        efficiency = (sum(s['reconciled'] for s in outputs) / sum(s['reconciled'] for s in inputs)) * 100
        
        # Save reconciliation summary
        cursor.execute('''
            INSERT INTO reconciliation_history 
            (timestamp, original_imbalance, final_imbalance, chi_square_statistic, 
             degrees_of_freedom, p_value, efficiency, streams_data, reconciled_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            original_imbalance,
            final_imbalance,
            self.reconciled_data['chi_square_statistic'],
            self.reconciled_data['degrees_of_freedom'],
            self.reconciled_data['p_value'],
            efficiency,
            json.dumps(self.streams),
            json.dumps(self.reconciled_data)
        ))
        
        # Save individual stream performance
        for stream in self.streams:
            cursor.execute('''
                INSERT INTO stream_performance
                (stream_name, timestamp, measured_value, reconciled_value, 
                 uncertainty, adjustment, normalized_residual)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                stream['name'],
                timestamp,
                stream['measured'],
                stream['reconciled'],
                stream['uncertainty'],
                stream['adjustment'],
                stream['normalized_residual']
            ))
        
        conn.commit()
        conn.close()
    
    def get_historical_analysis(self, days=30):
        """Get historical performance analysis"""
        conn = sqlite3.connect(self.db_path)
        
        # Get reconciliation history
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        history_df = pd.read_sql_query('''
            SELECT * FROM reconciliation_history 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', conn, params=[cutoff_date])
        
        # Get stream performance trends
        performance_df = pd.read_sql_query('''
            SELECT * FROM stream_performance 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', conn, params=[cutoff_date])
        
        conn.close()
        
        return {
            'reconciliation_history': history_df.to_dict('records'),
            'stream_performance': performance_df.to_dict('records')
        }
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.reconciled_data:
            return {"error": "No reconciliation data available"}
        
        inputs = [s for s in self.streams if s['type'] == 'input']
        outputs = [s for s in self.streams if s['type'] == 'output']
        
        report = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'total_streams': len(self.streams),
                'input_streams': len(inputs),
                'output_streams': len(outputs),
                'data_consistency': self.reconciled_data['data_consistency'],
                'chi_square_statistic': self.reconciled_data['chi_square_statistic'],
                'p_value': self.reconciled_data['p_value']
            },
            'mass_balance': {
                'original_imbalance': sum(s['measured'] for s in inputs) - sum(s['measured'] for s in outputs),
                'final_imbalance': sum(s['reconciled'] for s in inputs) - sum(s['reconciled'] for s in outputs),
                'closure_error': abs(sum(s['reconciled'] for s in inputs) - sum(s['reconciled'] for s in outputs)),
                'efficiency': (sum(s['reconciled'] for s in outputs) / sum(s['reconciled'] for s in inputs)) * 100
            },
            'stream_analysis': [],
            'recommendations': []
        }
        
        # Analyze each stream
        for stream in self.streams:
            analysis = {
                'name': stream['name'],
                'type': stream['type'],
                'measured': stream['measured'],
                'reconciled': stream['reconciled'],
                'adjustment': stream['adjustment'],
                'adjustment_percent': (abs(stream['adjustment']) / stream['measured']) * 100 if stream['measured'] != 0 else 0,
                'normalized_residual': stream['normalized_residual'],
                'status': stream['status'],
                'reliability_score': max(0, 100 - abs(stream['normalized_residual']) * 50)
            }
            report['stream_analysis'].append(analysis)
        
        # Generate recommendations
        error_streams = [s for s in self.streams if s['status'] == 'error']
        warning_streams = [s for s in self.streams if s['status'] == 'warning']
        
        if error_streams:
            report['recommendations'].append({
                'priority': 'HIGH',
                'type': 'Calibration',
                'message': f"Critical: {len(error_streams)} streams require immediate calibration check",
                'affected_streams': [s['name'] for s in error_streams]
            })
        
        if warning_streams:
            report['recommendations'].append({
                'priority': 'MEDIUM',
                'type': 'Monitoring',
                'message': f"Monitor {len(warning_streams)} streams for trending issues",
                'affected_streams': [s['name'] for s in warning_streams]
            })
        
        if self.reconciled_data['p_value'] < 0.05:
            report['recommendations'].append({
                'priority': 'HIGH',
                'type': 'Data Quality',
                'message': "Statistical test indicates potential systematic errors in measurements",
                'details': f"Chi-square p-value: {self.reconciled_data['p_value']:.4f}"
            })
        
        return report

# Flask API for integration with frontend
app = Flask(__name__)
CORS(app)

@app.route('/api/reconcile', methods=['POST'])
def reconcile_streams():
    """API endpoint for stream reconciliation"""
    try:
        data = request.json
        streams = data.get('streams', [])
        
        reconciler = AdvancedMassBalanceReconciler()
        reconciler.load_streams(streams)
        reconciler.weighted_least_squares_reconciliation()
        
        return jsonify({
            'success': True,
            'streams': reconciler.streams,
            'reconciliation_data': reconciler.reconciled_data,
            'report': reconciler.generate_performance_report()
        })
    
    except Exception as e:
        logger.error(f"Reconciliation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history/<int:days>', methods=['GET'])
def get_history(days):
    """Get historical analysis data"""
    try:
        reconciler = AdvancedMassBalanceReconciler()
        history = reconciler.get_historical_analysis(days)
        return jsonify({'success': True, 'data': history})
    
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export/<format>', methods=['POST'])
def export_data(format):
    """Export reconciliation data in various formats"""
    try:
        data = request.json
        
        if format == 'excel':
            # Create Excel file with multiple sheets
            with pd.ExcelWriter('mass_balance_report.xlsx', engine='openpyxl') as writer:
                # Streams data
                streams_df = pd.DataFrame(data.get('streams', []))
                streams_df.to_excel(writer, sheet_name='Streams', index=False)
                
                # Historical data if available
                reconciler = AdvancedMassBalanceReconciler()
                history = reconciler.get_historical_analysis(30)
                
                if history['reconciliation_history']:
                    history_df = pd.DataFrame(history['reconciliation_history'])
                    history_df.to_excel(writer, sheet_name='History', index=False)
                
                if history['stream_performance']:
                    performance_df = pd.DataFrame(history['stream_performance'])
                    performance_df.to_excel(writer, sheet_name='Performance', index=False)
            
            return send_file('mass_balance_report.xlsx', as_attachment=True)
        
        elif format == 'pdf':
            # Generate PDF report (requires additional libraries like reportlab)
            return jsonify({'success': False, 'error': 'PDF export not implemented yet'})
        
        else:
            return jsonify({'success': False, 'error': 'Unsupported format'}), 400
    
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Utility functions for advanced calculations
class ProcessOptimization:
    """Advanced process optimization utilities"""
    
    @staticmethod
    def calculate_energy_balance(streams, energy_data):
        """Calculate energy balance alongside mass balance"""
        # Implementation for energy balance calculations
        pass
    
    @staticmethod
    def monte_carlo_uncertainty(streams, n_simulations=1000):
        """Monte Carlo simulation for uncertainty propagation"""
        results = []
        
        for _ in range(n_simulations):
            # Generate random samples based on uncertainties
            simulated_values = []
            for stream in streams:
                noise = np.random.normal(0, stream['uncertainty'])
                simulated_values.append(stream['measured'] + noise)
            
            # Calculate balance for this simulation
            inputs = [simulated_values[i] for i, s in enumerate(streams) if s['type'] == 'input']
            outputs = [simulated_values[i] for i, s in enumerate(streams) if s['type'] == 'output']
            balance = sum(inputs) - sum(outputs)
            
            results.append(balance)
        
        return {
            'mean_imbalance': np.mean(results),
            'std_imbalance': np.std(results),
            'confidence_95': np.percentile(results, [2.5, 97.5]).tolist()
        }
    
    @staticmethod
    def detect_gross_errors(streams, threshold=3.0):
        """Detect gross errors in measurement data"""
        errors = []
        
        for i, stream in enumerate(streams):
            if hasattr(stream, 'normalized_residual'):
                if abs(stream['normalized_residual']) > threshold:
                    errors.append({
                        'stream_index': i,
                        'stream_name': stream['name'],
                        'normalized_residual': stream['normalized_residual'],
                        'severity': 'HIGH' if abs(stream['normalized_residual']) > 4.0 else 'MEDIUM'
                    })
        
        return errors

if __name__ == '__main__':
    # Example usage
    print("Advanced Mass Balance Reconciliation System")
    print("==========================================")
    
    # Sample data
    sample_streams = [
        {'id': 1, 'name': 'Crude Feed', 'type': 'input', 'measured': 1000, 'uncertainty': 20},
        {'id': 2, 'name': 'Steam', 'type': 'input', 'measured': 50, 'uncertainty': 5},
        {'id': 3, 'name': 'Gasoline', 'type': 'output', 'measured': 420, 'uncertainty': 15},
        {'id': 4, 'name': 'Diesel', 'type': 'output', 'measured': 350, 'uncertainty': 12},
        {'id': 5, 'name': 'Heavy Oil', 'type': 'output', 'measured': 200, 'uncertainty': 10},
        {'id': 6, 'name': 'Gas', 'type': 'output', 'measured': 60, 'uncertainty': 8},
        {'id': 7, 'name': 'Losses', 'type': 'output', 'measured': 15, 'uncertainty': 5}
    ]
    
    # Perform reconciliation
    reconciler = AdvancedMassBalanceReconciler()
    reconciler.load_streams(sample_streams)
    reconciler.weighted_least_squares_reconciliation()
    
    # Generate report
    report = reconciler.generate_performance_report()
    print(json.dumps(report, indent=2))
    
    # Start Flask API server
    print("\nStarting Flask API server...")
    print("Access API at: http://localhost:5000")
    print("Example endpoints:")
    print("  POST /api/reconcile - Perform reconciliation")
    print("  GET /api/history/30 - Get 30-day history")
    print("  POST /api/export/excel - Export to Excel")
    
    app.run(debug=True, host='0.0.0.0', port=5001)