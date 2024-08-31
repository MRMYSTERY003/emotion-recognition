from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return "server running"

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return 'No audio file found', 400

    audio_file = request.files['audio']
    audio_file.save('./audio/test.wav')
    print("audio received")
    return 'Audio file saved successfully!', 200

if __name__ == '__main__':
    app.run(debug=True)
