from flask import Flask, render_template, request, jsonify
import numpy as np
import json

app = Flask(__name__)

# Sample botanical database mapping
PLANT_DATABASE = {
    "Neem": {
        "scientific_name": "Azadirachta indica",
        "uses": "Antibacterial, treats skin conditions, promotes oral hygiene.",
        "active_compounds": "Nimbin, Azadirachtin"
    },
    "Tulsi": {
        "scientific_name": "Ocimum sanctum",
        "uses": "Relieves respiratory distress, boosts immunity, reduces stress.",
        "active_compounds": "Eugenol, Ursolic acid"
    },
    "Aloe_Vera": {
        "scientific_name": "Aloe barbadensis miller",
        "uses": "Soothes burns, aids digestive health, anti-inflammatory.",
        "active_compounds": "Aloin, Acemannan"
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/plants", methods=["GET"])
def get_plants():
    plants = [{"name": name, **info} for name, info in PLANT_DATABASE.items()]
    return jsonify(plants)

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Mock inference response for demonstration
    predicted_plant = "Neem"
    details = PLANT_DATABASE[predicted_plant]
    
    return jsonify({
        "status": "success",
        "prediction": predicted_plant,
        "confidence": 96.8,
        "details": details
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)