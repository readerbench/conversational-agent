const SpeechRecognition = window.webkitSpeechRecognition;

class SpeechToText {
    constructor(onSpeechResult, onAudioStateChanged) {
        let recognition = new SpeechRecognition();
        this.recognition = recognition;
        recognition.lang = 'ro-RO';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onresult = function (event) {
            const speechResult = event.results[0][0].transcript.toLowerCase();
            const confidence = event.results[0][0].confidence.toFixed(2);
            onSpeechResult(speechResult, confidence);
        };

        recognition.onspeechend = function () {
            console.log('SpeechRecognition.onspeechend');
            recognition.stop();
        };

        recognition.onerror = function (event) {
        };

        recognition.onaudiostart = function (event) {
            // Fired when the user agent has started to capture audio.
            console.log('SpeechRecognition.onaudiostart');
            onAudioStateChanged();
        };

        recognition.onaudioend = function (event) {
            // Fired when the user agent has finished capturing audio.
            console.log('SpeechRecognition.onaudioend');
            recognition.stop();
            onAudioStateChanged();
        };

        recognition.onend = function (event) {
            // Fired when the speech recognition service has disconnected.
            console.log('SpeechRecognition.onend');
        };

        recognition.onnomatch = function (event) {
            // Fired when the speech recognition service returns a final result with no significant recognition.
            // This may involve some degree of recognition, which doesn't meet or exceed the confidence threshold.
            console.log('SpeechRecognition.onnomatch');
        };

        recognition.onsoundstart = function (event) {
            // Fired when any sound — recognisable speech or not — has been detected.
            console.log('SpeechRecognition.onsoundstart');
        };

        recognition.onsoundend = function (event) {
            // Fired when any sound — recognisable speech or not — has stopped being detected.
            console.log('SpeechRecognition.onsoundend');
        };

        recognition.onspeechstart = function (event) {
            // Fired when sound that is recognised by the speech recognition service as speech has been detected.
            console.log('SpeechRecognition.onspeechstart');
        };

        recognition.onstart = function (event) {
            // Fired when the speech recognition service has begun listening to incoming audio with intent to
            // recognize grammars associated with the current SpeechRecognition.
            console.log('SpeechRecognition.onstart');
        };
    }

    execute() {
        this.recognition.start();
    }
}

export default SpeechToText;
