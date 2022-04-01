const phrase = document.getElementById('phrase');
const phraseRow = document.getElementById('phrase_row');
const idRow = document.getElementById('id_row');
const result = document.getElementById('deps');

const dependencies = [
    "-",
    "prep",
    "ROOT",
    "cine",
    "care",
    "ce",
    "pe cine",
    "unde",
    "când",
    "cât timp",
    "cât de des",
    "la cât timp",
    "cum este",
    "ce este",
    "care este",
    "al cui",
    "ce fel de",
    "cât",
    "cui",
];

let parseRes = {
    heads: [],
    deps: []
};

let currentToken;
let pairToken;
let tokenButtons;

function init() {
    phrase.innerText = "";
    indexTokens();

    const depsList = document.getElementById("deps_list");

    for (const dep of dependencies) {
        depsList.innerHTML += `<button class="dep_btn" onclick="depSelected(this.id)" id="${dep}">${dep}</button>`;
    }
}

function indexTokens() {
    parseRes.heads = [];
    parseRes.deps = [];
    requestSpacyDepParse();
    tokenButtons = [];

    // clear table
    phraseRow.innerHTML = idRow.innerHTML = '';

    const tokens = phrase.value.split(/[ ]/);
    tokens.forEach((token, i) => {
        phraseRow.innerHTML += `<td align="center"><button id="${i}" onclick="tokenSelected(this.id)" class="btn">${token}</button></td>`;
        idRow.innerHTML += `<td align="center">${i}</td>`;
    });

    result.value = '"heads": [],\n"deps": [],\n';

    interactiveAnnotate(tokens);
}

function tokenSelected(id) {
    pairToken = parseInt(id);

    tokenButtons[pairToken].className = "btn-blue";
}

function depSelected(dep) {
    parseRes.heads.push(pairToken);
    parseRes.deps.push(dep);

    tokenButtons[pairToken].className = "btn";

    if (currentToken < tokenButtons.length - 1) {
        // go to next token
        tokenButtons[currentToken].className = "btn";
        currentToken++;
        tokenButtons[currentToken].className = "btn-sel";
    } else {
        tokenButtons[currentToken].className = "btn";
        parent.focus();
    }

    const data = JSON.stringify(parseRes)
        .replace(/(("heads")|("deps"))/g, '\n  $1')
        .replace('}', '\n}');
    result.value = `[\n"${phrase.value}",\n${data}\n]`;
}

function interactiveAnnotate(tokens) {
    for (let i = 0; i < tokens.length; i++) {
        tokenButtons.push(document.getElementById(`${i}`));
    }

    currentToken = 0;
    tokenButtons[currentToken].className = "btn-sel";
}

function requestSpacyDepParse() {
    const xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () {
        if (this.readyState === 4 && this.status === 200) {
            let iframe = document.getElementById('embed');
            iframe.src = iframe.src;
        }
    };
    xhttp.open("POST", "http://localhost:3333", true);
    xhttp.setRequestHeader("Access-Control-Allow-Origin", 'http://localhost:3333/');
    xhttp.setRequestHeader("Access-Control-Allow-Methods", "GET, PUT, POST, DELETE, HEAD, OPTIONS");
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send(phrase.value);
}

init();