import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
from predict import (
    predict_crop_single, predict_crop_batch,
    predict_fertility_single, predict_fertility_batch
)

# ==================== CONFIG ====================
UPLOAD_FOLDER = "uploads"
IMG_FOLDER = "static/img"
ALLOWED_EXT = {'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'super-secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.filters['zip'] = zip  # Enable Jinja2 zip filter

DB = 'users.db'

# ==================== DATABASE ====================
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )''')
    conn.commit()
    conn.close()


def create_user(username, password):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()


def get_user(username):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT id, username, password FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row


# ==================== HELPERS ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def save_charts(chart_info, fertility_level=None):
    labels = chart_info['labels']
    values = chart_info['values']

    plt.figure(figsize=(8, 4))

    # Color scheme based on fertility
    if fertility_level and fertility_level.lower() == "high":
        colors = ['#2ecc71'] * len(labels)
    elif fertility_level and fertility_level.lower() == "medium":
        colors = ['#f1c40f'] * len(labels)
    elif fertility_level and fertility_level.lower() == "low":
        colors = ['#e74c3c'] * len(labels)
    else:
        colors = ['#3498db'] * len(labels)

    bars = plt.bar(labels, values, color=colors)
    plt.title("Soil Feature Levels")
    plt.ylabel("Value")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=8)

    plt.savefig(os.path.join(IMG_FOLDER, "bar.png"))
    plt.close()


# ==================== ROUTES ====================

@app.route('/')
def home():
    return render_template('index.html')


# ---------- SIGNUP WITH PASSWORD VALIDATION ----------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        confirm = request.form['confirm']

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return redirect(url_for('signup'))
        if not any(char.isdigit() for char in password):
            flash('Password must contain at least one number.', 'danger')
            return redirect(url_for('signup'))
        if not any(char.isupper() for char in password):
            flash('Password must contain at least one uppercase letter.', 'danger')
            return redirect(url_for('signup'))
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('signup'))

        try:
            hashed = generate_password_hash(password)
            create_user(username, hashed)
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception:
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))

    return render_template('signup.html')


# ---------- LOGIN ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['user'] = {'id': user[0], 'username': user[1]}
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
        return redirect(url_for('login'))
    return render_template('login.html')


# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))


# ==================== DASHBOARD ====================
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        flash('Please login first.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Manual input
        if 'manual' in request.form:
            try:
                N = float(request.form['N'])
                P = float(request.form['P'])
                K = float(request.form['K'])
                pH = float(request.form['pH'])
                temp = float(request.form['temperature'])
                moisture = float(request.form['moisture'])

                crop = predict_crop_single(N, P, K, pH, temp, moisture)
                fertility = predict_fertility_single(N, P, K, pH, temp, moisture)

                chart_info = {
                    'labels': ['N', 'P', 'K', 'pH', 'Temperature', 'Moisture'],
                    'values': [N, P, K, pH, temp, moisture]
                }
                save_charts(chart_info, fertility)

                # Fertility recommendation
                if fertility.lower() == "high":
                    suggestion = {
                        "icon": "🟢",
                        "level": "High Fertility",
                        "message": "Your soil is rich and nutrient-dense. Maintain balance using organic manure and mild NPK (10:26:26).",
                        "fertilizer": "Recommended: Organic compost, 10:26:26 NPK, and controlled urea application."
                    }
                elif fertility.lower() == "medium":
                    suggestion = {
                        "icon": "🟡",
                        "level": "Medium Fertility",
                        "message": "Your soil is moderately fertile. Apply balanced fertilizers before sowing.",
                        "fertilizer": "Recommended: 12:32:16 NPK blend or 17:17:17 compound fertilizer."
                    }
                else:
                    suggestion = {
                        "icon": "🔴",
                        "level": "Low Fertility",
                        "message": "Your soil fertility is low. Improve using compost, biofertilizers, and micronutrients.",
                        "fertilizer": "Recommended: Organic manure + 20:20:0 NPK + micronutrient mix."
                    }

                session['result_data'] = {
                    'mode': 'manual',
                    'input': chart_info,
                    'crop': crop,
                    'fertility': fertility,
                    'suggestion': suggestion
                }

                return redirect(url_for('results'))

            except Exception as e:
                flash(f"Error: {e}", 'danger')

        # CSV Upload with farm-level aggregation
        elif 'csvfile' in request.files:
            f = request.files['csvfile']
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(path)
                try:
                    df = pd.read_csv(path)
                    required = {'N', 'P', 'K', 'pH', 'temperature', 'moisture'}
                    if not required.issubset(df.columns):
                        flash('CSV missing required columns.', 'danger')
                    else:
                        out = predict_crop_batch(df)
                        out = predict_fertility_batch(out)

                        most_common_crop = out['predicted_crop'].mode()[0]
                        avg_values = df[['N', 'P', 'K', 'pH', 'temperature', 'moisture']].mean().to_dict()

                        chart_info = {
                            'labels': list(avg_values.keys()),
                            'values': list(avg_values.values())
                        }
                        save_charts(chart_info)

                        summary = f"Based on {len(df)} soil samples, the recommended crop for your farm is {most_common_crop}. " \
                                  f"The soil shows a balanced fertility pattern suitable for {most_common_crop} cultivation."

                        session['result_data'] = {
                            'mode': 'csv',
                            'file': f'predictions_{filename}',
                            'sample': out.head(5).to_dict(orient='records'),
                            'input': chart_info,
                            'crop': most_common_crop,
                            'fertility': 'Mixed / Averaged',
                            'summary': summary
                        }

                        return redirect(url_for('results'))
                except Exception as e:
                    flash(f"Error processing CSV: {e}", 'danger')
            else:
                flash('Invalid or no file uploaded.', 'warning')

    return render_template('dashboard.html')


# ==================== RESULTS ====================
@app.route('/results')
def results():
    if 'result_data' not in session:
        flash('No results to display. Please submit data first.', 'warning')
        return redirect(url_for('dashboard'))
    return render_template('results.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


# ==================== MAIN ====================
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
