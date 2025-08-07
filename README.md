# 🕳️ Pothole Management System

## 🧠 Overview

The **Pothole Management System** is a smart, AI-enabled web application built to detect and manage potholes using deep learning and a role-based complaint management system. The platform supports:

- CNN-based pothole classification  
- User/Admin login with OTP verification  
- Image upload and automated classification  
- Jurisdiction-based admin dashboards  
- Complaint tracking, verification, and analytics

---

## 🚀 Features

- 🧠 Detects `Pothole`, `Normal`, and `Invalid` road images using CNN  
- 📷 Users can upload complaints with images and messages  
- 🔐 Secure login with role selection and OTP verification  
- 🗂️ Admins view complaints based on their jurisdiction only  
- 📊 Visual dashboards with bar, line, pie charts, and flashcards  
- 🧾 Complaint verification and status update system  
- 🗃️ Persistent data storage using SQLite  

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, Bootstrap, Chart.js  
- **Backend**: Python (Flask)  
- **Deep Learning**: TensorFlow/Keras  
- **Database**: SQLite (via SQLAlchemy)  
- **Libraries**:  
  - `opencv-python`, `Pillow`  
  - `tensorflow`, `keras`  
  - `sklearn`, `flask_sqlalchemy`  

---

## 📁 Project Structure

```
PotholeManagementSystem/
├── app.py                     # Flask app with all routes & logic
├── sample.h5                  # Trained CNN model
├── pothole.db                 # SQLite database
├── static/
│   └── uploads/               # Uploaded images
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── otp.html
│   ├── complain.html
│   ├── complain_status.html
│   ├── admin_dashboard.html
│   ├── admin_issues.html
│   └── admin_dashboard_charts.html
└── model_training.py          # CNN training and evaluation code
```

---

## 🔐 Admin Credentials

| Jurisdiction         | Username         | Password     |
|----------------------|------------------|--------------|
| Central Delhi        | `centraladmin`   | `central123` |
| East Delhi           | `eastadmin`      | `east123`    |
| New Delhi            | `newadmin`       | `new123`     |
| North Delhi          | `northadmin`     | `north123`   |
| North East Delhi     | `northeastadmin` | `ne123`      |
| North West Delhi     | `northwestadmin` | `nw123`      |
| Shahdara             | `shahdaraadmin`  | `shah123`    |
| South Delhi          | `southadmin`     | `south123`   |
| South East Delhi     | `southeastadmin` | `se123`      |
| South West Delhi     | `southwestadmin` | `sw123`      |
| West Delhi           | `westadmin`      | `west123`    |

🔑 OTP for all users: `1234`

---

## 📸 Complaint Flow

1. **User Login** → Select Role and Jurisdiction  
2. **OTP Verification** (Fixed OTP: `1234`)  
3. **Submit Complaint** with location, image, and message  
4. **Image Prediction** using CNN model:  
   `Pothole`, `Normal`, or `Invalid`  
5. **Admin Panel**: View, Verify, Update Status  
6. **Analytics Dashboard**: Visualize complaints

---

## 📊 Dashboard Overview

**Admin dashboard includes:**

- ✅ Pie chart of verification status (Verified / Not Verified / Rejected)  
- 📅 Bar chart of complaints by month  
- 📈 Line chart of complaints by year  
- 🔢 Flashcard showing total complaints  
- ⚒️ Bar chart of complaint status (`Under Process` / `Pothole Fixed`)  

---

## 🧪 CNN Model Training

- Grayscale images resized to 100x100  
- 3 classes: `Pothole`, `Plain`, `Invalid`  
- ImageDataGenerator used for augmentation  
- Final model saved as `sample.h5`  

**Training Script Highlights:**

```python
Conv2D(32) → Conv2D(64)  
→ GlobalAveragePooling2D  
→ Dense(128) → Dropout(0.3)  
→ Dense(3) with softmax
```

> Accuracy achieved: ~90%  
> Loss Function: `categorical_crossentropy`  
> Optimizer: `adam`

---

## 💻 Running Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Pothole-Management-System.git
cd Pothole-Management-System
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
python app.py
```

> Open your browser and go to `http://127.0.0.1:5000/`

---

## 🔮 Future Enhancements

- 📸 Auto-extract GPS & timestamp from image metadata  
- 📱 Convert to a responsive PWA for mobile use  
- ✉️ Email notifications to users & admins  
- 🧩 Microservices for model and database components  
- 🔐 Enhanced security with password hashing and 2FA  
- 📊 Time-series forecasting of pothole occurrences  

---

## 🙌 Contributions

Pull requests are welcome. Please open an issue to discuss any major feature or refactor you plan to introduce.

---
