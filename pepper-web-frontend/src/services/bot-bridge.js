import axios from 'axios';

const API_HOST = process.env.RASA_SERVER_HOST;
const BOT_ENDPOINT = `${API_HOST}/webhooks/rest/webhook`;
const PREDICT_ENDPOINT = `${API_HOST}/conversations/0/messages`;

export const sendMsgAndGetReply = async (msg) => {
    const config = {
        headers: {
            'Access-Control-Allow-Origin': '*',
        }
    };

    const [response, prediction] = await Promise.all([
        axios.post(BOT_ENDPOINT, {
            sender: "anonymus",
            message: msg
        }, config),
        axios.post(PREDICT_ENDPOINT, {
            sender: "user",
            text: msg
        }, config)]);

    return [
        response.data?.[0]?.text || "Am întâmpinat o eroare 😥",
        prediction.data.latest_message.intent
    ];
};