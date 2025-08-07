from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pothole.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)

model = load_model('sample.h5')


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    jurisdiction = db.Column(db.String(50), nullable=True)


class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User', backref=db.backref('complaints', lazy=True))
    location = db.Column(db.String(100))
    nearest_iconic_place = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(15))
    date = db.Column(db.String(20))
    message = db.Column(db.Text)
    photo = db.Column(db.String(100))
    status = db.Column(db.String(30), default='underprocess')
    prediction = db.Column(db.String(20))
    user_verified = db.Column(db.String(10))


with app.app_context():
    db.create_all()

    # ✅ Fixed admin credentials by jurisdiction
    admin_credentials = {
        "Central Delhi": ("centraladmin", "central123"),
        "East Delhi": ("eastadmin", "east123"),
        "New Delhi": ("newadmin", "new123"),
        "North Delhi": ("northadmin", "north123"),
        "North East Delhi": ("northeastadmin", "ne123"),
        "North West Delhi": ("northwestadmin", "nw123"),
        "Shahdara": ("shahdaraadmin", "shah123"),
        "South Delhi": ("southadmin", "south123"),
        "South East Delhi": ("southeastadmin", "se123"),
        "South West Delhi": ("southwestadmin", "sw123"),
        "West Delhi": ("westadmin", "west123")
    }

    # Create default user
    if not User.query.filter_by(username='Albin').first():
        db.session.add(User(username='Albin', password='1234', role='user'))

    # Create fixed admin users
    for jurisdiction, (username, password) in admin_credentials.items():
        if not User.query.filter_by(username=username).first():
            db.session.add(User(username=username, password=password, role='admin', jurisdiction=jurisdiction))

    db.session.commit()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        jurisdiction = request.form.get('jurisdiction')

        if role == 'admin':
            user = User.query.filter_by(username=username, password=password, role=role, jurisdiction=jurisdiction).first()
        else:
            user = User.query.filter_by(username=username, password=password, role=role).first()

        if user:
            session['temp_user'] = {
                'username': user.username,
                'role': user.role,
                'jurisdiction': user.jurisdiction if user.role == 'admin' else None
            }
            return redirect(url_for('otp_verify'))

        flash('Invalid credentials or jurisdiction mismatch')
    return render_template('login.html')


@app.route('/otp', methods=['GET', 'POST'])
def otp_verify():
    if 'temp_user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        otp = request.form['otp']
        if otp == '1234':
            user_data = session.pop('temp_user')
            session['username'] = user_data['username']
            session['role'] = user_data['role']
            if user_data['role'] == 'admin':
                session['jurisdiction'] = user_data['jurisdiction']
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid OTP')
    return render_template('otp.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/user')
def user_dashboard():
    return render_template('user_dashboard.html')


@app.route('/complain', methods=['GET', 'POST'])
def complain():
    if request.method == 'POST':
        user = User.query.filter_by(username=session['username']).first()

        location = request.form['location']
        nearest_iconic_place = request.form['nearest_iconic_place']
        email = request.form['email']
        phone = request.form['phone']
        date = request.form['date']  # <-- updated line
        message = request.form['message']
        file = request.files['photo']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        new_complaint = Complaint(
            user=user,
            location=location,
            nearest_iconic_place=nearest_iconic_place,
            email=email,
            phone=phone,
            date=date,
            message=message,
            photo=filename
        )
        db.session.add(new_complaint)
        db.session.commit()
        return redirect(url_for('user_dashboard'))

    all_areas = list(admin_credentials.keys())
    if session['role'] == 'admin':
        locations = [session['jurisdiction']]
    else:
        locations = all_areas

    return render_template('complain.html', locations=locations)


@app.route('/complain_status', methods=['GET', 'POST'])
def complain_status():
    user = User.query.filter_by(username=session['username']).first()
    complaints = Complaint.query.filter_by(user=user).all()
    return render_template('complain_status.html', complaints=complaints)


@app.route('/verify/<int:id>', methods=['POST'])
def verify_complaint(id):
    complaint = Complaint.query.get_or_404(id)
    complaint.user_verified = request.form['verification']
    db.session.commit()
    return redirect(url_for('complain_status'))


@app.route('/admin')
def admin_dashboard():
    return render_template('admin_dashboard.html')


@app.route('/admin_issues')
def admin_issues():
    complaints = Complaint.query.filter_by(location=session['jurisdiction']).all()
    return render_template('admin_issues.html', complaints=complaints)


# @app.route('/predict/<int:id>', methods=['POST'])
# def predict_pothole(id):
#     complaint = Complaint.query.get_or_404(id)

#     # Load and preprocess image
#     img_path = os.path.join(app.config['UPLOAD_FOLDER'], complaint.photo)
#     image = Image.open(img_path).convert('L').resize((100, 100))
#     img_array = np.expand_dims(np.array(image) / 255.0, axis=(0, -1))  # shape: (1, 100, 100, 1)

#     # Predict
#     prediction = model.predict(img_array)[0]  # prediction is [normal_prob, pothole_prob]
#     normal_prob, pothole_prob = prediction[0], prediction[1]

#     print(f"Prediction probabilities: Normal={normal_prob}, Pothole={pothole_prob}")

#     # Apply confidence threshold to handle invalid images
#     if max(normal_prob, pothole_prob) < 0.7:  # You can adjust this threshold
#         predicted_label = 'invalid'
#     else:
#         predicted_label = 'pothole' if pothole_prob > normal_prob else 'normal'

#     # Save prediction
#     complaint.prediction = predicted_label
#     db.session.commit()

#     return redirect(url_for('admin_issues'))


@app.route('/predict/<int:id>', methods=['POST'])
def predict_pothole(id):
    from PIL import Image
    import numpy as np
    import os

    complaint = Complaint.query.get_or_404(id)

    # Load and preprocess image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], complaint.photo)
    image = Image.open(img_path).convert('L').resize((100, 100))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=(0, -1))  # shape: (1, 100, 100, 1)

    # Predict
    prediction = model.predict(img_array)[0]  # prediction is a list of 3 probabilities
    class_names = ['pothole', 'normal', 'invalid']
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]

    print(f"Prediction probabilities: Pothole={prediction[0]:.3f}, Normal={prediction[1]:.3f}, Invalid={prediction[2]:.3f}")
    print(f"Predicted label: {predicted_label}")

    # Save prediction
    complaint.prediction = predicted_label
    db.session.commit()

    return redirect(url_for('admin_issues'))




@app.route('/update_status/<int:id>', methods=['POST'])
def update_status(id):
    complaint = Complaint.query.get_or_404(id)
    complaint.status = request.form['status']
    db.session.commit()
    return redirect(url_for('admin_issues'))


@app.route('/admin_dashboard_charts')
def admin_dashboard_charts():
    complaints = Complaint.query.filter_by(location=session['jurisdiction']).all()
    verified = sum(1 for c in complaints if c.user_verified == 'yes')
    not_verified = sum(1 for c in complaints if c.user_verified is None)
    rejected = sum(1 for c in complaints if c.user_verified == 'no')
    verification_counts = [verified, not_verified, rejected]
    months = ['{:02d}'.format(i) for i in range(1, 13)]
    month_counts = [sum(1 for c in complaints if c.date and c.date.split('-')[1] == m) for m in months]
    years = sorted(set(c.date.split('-')[0] for c in complaints if c.date))
    year_counts = [sum(1 for c in complaints if c.date and c.date.split('-')[0] == y) for y in years]
    status_labels = ['Under Process', 'Pothole Fixed']
    status_counts = [
        sum(1 for c in complaints if c.status == 'underprocess'),
        sum(1 for c in complaints if c.status == 'pothole fixed')
    ]

    return render_template('admin_dashboard_charts.html',
        status_labels=status_labels,
        status_counts=status_counts,
        month_labels=months,
        month_counts=month_counts,
        verification_counts=verification_counts,
        year_labels=years,
        year_counts=year_counts,
        total_complaints=len(complaints)
    )


if __name__ == '__main__':
    app.run(debug=True)
