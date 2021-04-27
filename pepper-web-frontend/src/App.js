import React, {useState} from 'react';
import './App.css';
import SpeechToText from "./services/speech-to-text";
import {sendMsgAndGetReply} from "./services/bot-bridge";
import illustration from './res/illustration_1.png';
import bot from './res/bot.png';

function App() {
    const [messages, setMessages] = useState([]);

    const sendUserInput = (msg, speechToTextConfidence) => {
        if (!msg.trim().length)
            return;

        setMessages(messages => [
            ...messages,
            {
                author: "me",
                text: msg,
                metadata: speechToTextConfidence ? `speech-to-text confidence: ${speechToTextConfidence}` : ""
            }
        ]);

        setMessages(messages => [
            ...messages,
            {
                author: "bot",
                text: null
            }
        ]);

        sendMsgAndGetReply(msg).then(([reply, predictedIntent]) => {
            setMessages(messages => [
                ...messages.slice(0, -1),
                {
                    author: "bot",
                    text: reply,
                    metadata: `intent: ${predictedIntent.name}, confidence: ${predictedIntent.confidence.toFixed(2)}`
                }
            ]);
        });
    };

    return (
        <>
            <img src={illustration} className="illustration" alt="Bot illustration"/>
            <div className="container">
                <div className="header">
                    <p><span style={{color: "#cad9ea"}}></span>Conversational agent</p>
                </div>
                <div className="scrollable-pane">
                    <div className="wrapper">
                        {messages.map((msg, idx) => (
                            <MessageBubble author={msg.author} msg={msg.text} metadata={msg.metadata}
                                           key={idx}/>
                        ))}
                    </div>
                </div>
                <InputBox onUserMessage={sendUserInput}/>
            </div>
        </>
    );
}

const InputBox = (props) => {
    const [listeningToUser, setListeningToUser] = useState(false);

    const onUserVoiceInput = (msg, confidence) => {
        props.onUserMessage(msg, confidence);
    };

    const onAudioStateChanged = () => {
        setListeningToUser(listeningToUser => !listeningToUser);
    };

    const speechToTextConverter = new SpeechToText(onUserVoiceInput, onAudioStateChanged);

    const sendUserInput = () => {
        const inputTextField = document.getElementById("input");
        const message = inputTextField.value;
        inputTextField.value = "";

        props.onUserMessage(message);
    };

    document.onkeydown = e => {
        if (e.ctrlKey && e.key === "i")
            speechToTextConverter.execute();
    };

    return (
        <div className="input-box">
            <div className="spinner-grow" style={{visibility: listeningToUser ? "visible" : "hidden"}}/>
            <button className="icon-btn tooltip" onClick={() => speechToTextConverter.execute()}>
                <i className="fas fa-microphone"/>
                {/*<span className="tooltip-text">Utter a request</span>*/}
            </button>

            <input id="input" className="text-field" type="text"
                   placeholder="Type a message..."
                   autoComplete="off"
                   autoFocus
                   onKeyDown={e => {
                       if (e.key === "Enter") {
                           e.preventDefault();
                           sendUserInput();
                       }
                   }}
            />
            <button className="icon-btn tooltip" onClick={sendUserInput}>
                <i className="fas fa-paper-plane"/>
                {/*<span className="tooltip-text">Send</span>*/}
            </button>
        </div>
    );
};

const MessageBubble = (props) => {
    const MsgFormatter = (props) => {
        const lines = props.msg.split('\n');
        if (lines.length <= 1)
            return lines;

        return lines.map((line, idx) => {
            const tokens = line.split('➜');
            return <div key={idx}>
                <span className="emphasized-text">{tokens[0]}</span>
                <br/>
                <span>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    {tokens[1]}
                </span>
                <br/>
            </div>;
        });
    };

    return (
        <div className="stack-layer">
            {
                props.metadata &&
                <p className="bubble-metadata" style={{textAlign: props.author === "me" ? "right" : "left"}}>
                    {props.metadata}
                </p>
            }
            <div>
                {
                    props.author === "bot" &&
                    <img src={bot} className="persona" alt="bot avatar"
                         style={{textAlign: props.author === "me" ? "right" : "left"}}/>
                }
                <div className={"bubble " + (props.author === "me" ? "bubble-right" : "bubble-left")}>
                    {
                        props.msg
                            ? <MsgFormatter msg={props.msg}/>
                            : <TypingIndicator/>
                    }
                </div>
            </div>
        </div>
    );
};

const TypingIndicator = () => (
    <div className="ticontainer">
        <div className="tiblock">
            <div className="tidot"/>
            <div className="tidot"/>
            <div className="tidot"/>
        </div>
    </div>
);

export default App;
