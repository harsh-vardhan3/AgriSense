import firebase_admin
from firebase_admin import credentials, db

def call_firebase():
    # Initialize Firebase Admin SDK with the service account JSON file
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://cropp-cbe63-default-rtdb.firebaseio.com'  # Your Firebase Realtime Database URL
        })

    # Reference to the Firebase Realtime Database path
    ref = db.reference('/data2')

    # Retrieve the latest data based on the key (or timestamp)
    latest_data = ref.order_by_key().limit_to_last(1).get()

    if latest_data:
        # Print the latest data
        first_key = list(latest_data.keys())[0]
        specific_data = latest_data[first_key]

        # Extract data with default values in case of missing keys
        nitrogen = specific_data.get('nitrogen', 0)
        phosphorus = specific_data.get('phosphorus', 0)
        potassium = specific_data.get('potassium', 0)
        soilMoisture = specific_data.get('soilMoisture', 0)
        soilPH = specific_data.get('soilPH', 0)

        # Return data as a list
        return [nitrogen, phosphorus, potassium, soilMoisture, soilPH]
    
    return None  # Return None if no data is found
