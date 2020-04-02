window.onload = function (e) {
    var div = document.getElementById('chatBox');
    div.scrollTop = div.scrollHeight - div.clientHeight;

    var divd = document.getElementById('div1');
    myAddEvent(divd, 'click', donghua(divd));

    // document.getElementById('loading-answer').style.bottom = document.getElementById('input-div').clientHeight
};

function myAddEvent(obj, event, fn) {
    if (obj.attachEvent) {
        obj.attachEvent('on' + 'event', fn);
    } else {
        obj.addEventListener(event, fn, false);
    }
};

// image slideshow
function donghua(divd) {
    var oUl = divd.getElementsByTagName('ul')[0];
    var aLi = oUl.getElementsByTagName('li');
    oUl.innerHTML = oUl.innerHTML + oUl.innerHTML + oUl.innerHTML;
    oUl.style.width = aLi[0].offsetWidth * aLi.length + 40 + 'px';

    window.setInterval(function () {
        if (oUl.offsetLeft < -oUl.offsetWidth / 3) {
            oUl.style.left = '0';
        }
        if (oUl.offsetLeft > 0) {
            oUl.style.left = -oUl.offsetWidth / 2 + 'px';
        }
        oUl.style.left = oUl.offsetLeft - 3 + 'px';
    }, 30);
};

const chat_window = document.getElementById('chatBox')
const textBox = document.getElementsByName('context')[0]
const enter_button = document.getElementsByName('enter-button')[0]

// add question, answer pair to databse
function addToDb(question, answer, question_date, answer_date) {
    fetch('/add-sentence-to-db', {
        method: 'post',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: question, answer: answer, question_date: question_date, answer_date: answer_date })
    })
        .then(response => { return response.text(); })
        .then(text => {
            console.log("Chatbot submit " + text);
        });
}

// adds user input sentence to database and appends to chatbot window 
function processUserInput() {
    // get date and time of user question submitted
    const d = new Date();
    let date = d.toISOString().split('T')[0];
    let time = d.toLocaleString('en-GB').split(',')[1].trim();

    // clone template html message div and set text value
    let textbox = document.getElementsByName('context')[0];
    let sentence = textbox.value;
    const user_sentence_div = document.getElementById('user_sentence_template');
    const clone = user_sentence_div.cloneNode(true);
    clone.style.display = 'block';
    clone.childNodes[0].nextSibling.childNodes[0].nextSibling.textContent = sentence;
    clone.childNodes[0].nextSibling.childNodes[4].nextSibling.nextSibling.textContent = date + ' ' + time;

    textbox.value = '';

    const append_sentences_div = document.getElementById('appended_sentences');
    append_sentences_div.appendChild(clone);

    const loading = document.getElementById('loading-answer');
    loading.style.display = 'block';
    chat_window.scrollTop = chat_window.scrollHeight - chat_window.clientHeight;

    let date1;
    let time1;

    // get backend algorithm answer
    fetch('/query', {
        method: 'post',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: sentence })
    })
        .then(response => { return response.json() })
        .then(json => {
            if (json.answer === "I don't know.") {
                return false;
            }
            const answer = document.getElementById('bot_sentence_template').cloneNode(true);
            answer.style.display = 'block';

            const d1 = new Date();
            date1 = d1.toISOString().split('T')[0];
            time1 = d1.toLocaleString('en-GB').split(',')[1].trim();
            answer.childNodes[0].nextSibling.childNodes[2].nextSibling.textContent = json['answer'];
            answer.childNodes[0].nextSibling.childNodes[4].nextSibling.nextSibling.textContent = date1 + ' ' + time1;

            const div = document.getElementById('appended_sentences');
            div.appendChild(answer);

            // scroll chat window back to bottom after appending
            loading.style.display = 'none';
            chat_window.scrollTop = chat_window.scrollHeight - chat_window.clientHeight;
            return json['answer'];
        })
        .then(answer => {
            if (answer) {
                // store backend algorithm answer
                if (!date1 || !time1) {
                    date1 = date;
                    time1 = time;
                }
                addToDb(sentence, answer, date + ' ' + time, date1 + ' ' + time1);

            } else {
                // get Dialogflow answer
                fetch('/send_message', {
                    method: 'post',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: sentence })
                })
                    .then(response => { return response.json() })
                    .then(json => {
                        const answer = document.getElementById('bot_sentence_template').cloneNode(true);
                        answer.style.display = 'block';

                        const d1 = new Date();
                        date1 = d1.toISOString().split('T')[0];
                        time1 = d1.toLocaleString('en-GB').split(',')[1].trim();

                        answer.childNodes[0].nextSibling.childNodes[2].nextSibling.textContent = json['message'];
                        answer.childNodes[0].nextSibling.childNodes[4].nextSibling.nextSibling.textContent = date1 + ' ' + time1;

                        const div = document.getElementById('appended_sentences');
                        div.appendChild(answer);

                        // scroll chat window back to bottom after appending
                        loading.style.display = 'none';
                        chat_window.scrollTop = chat_window.scrollHeight - chat_window.clientHeight;
                        return json['message'];
                    })
                    .then(resp => {
                        addToDb(sentence, resp, date + ' ' + time, date1 + ' ' + time1);  // need dateTime  for both sentence and answer
                    })
                    .catch(err => { console.log(err) });
            }
        })
        .catch(error => {
            console.log(error);
        });
}

// user clicked Submit button after typing question
enter_button.addEventListener('click', () => {
    processUserInput();
});

// user pressed Enter after typing question
textBox.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        processUserInput();
    }
});

// change left side of the page to show Instructions, Feedback and ContactUs
document.getElementById('Home').addEventListener('click', () => {
    let home = document.getElementById('Home');
    let news_content = document.getElementById('home-content');
    let home_content = document.getElementById('div1');
    if (home.innerText === 'Display') {
        home.innerText = 'Home';
        news_content.style.display = 'none';
        home_content.style.display = 'block';

    } else {
        home.innerText = 'Display';
        home_content.style.display = 'none';
        news_content.style.display = 'block';
    }
});

// clear chat history when clear button is clicked
document.getElementById('clear-button').addEventListener('click', () => {
    // remove all sentences loaded from database
    let chatbot_window = document.getElementById('chatbot_window');
    while (chatbot_window.firstChild) {
        chatbot_window.removeChild(chatbot_window.firstChild);
    }

    // clear new sentences that were appended to the chat
    let appended_sentences = document.getElementById('appended_sentences');
    if (appended_sentences.hasChildNodes()) {
        while (appended_sentences.firstChild) {
            appended_sentences.removeChild(appended_sentences.firstChild);
        }
    }

    // remove sentences in database
    fetch('/clear-history', {
        method: 'post',
        headers: { 'Content-Type': 'application/json' }
    })
        .then(resp => {
            console.log("clear history OK")
        });
});


let index = 0;
let prevSearch = '';
let search_error_msg = document.getElementById('search-error');

// show search input box
document.getElementById('search-button').addEventListener('click', () => {
    document.getElementById('search-history').style.display = 'block';
});

// close search input box
document.getElementById('search-close').addEventListener('click', () => {
    index = 0;
    search_error_msg.innerText = '';
    search_error_msg.style.display = 'none';
    document.getElementById('search-history').style.display = 'none';
    document.getElementById('search').value = '';
});

// scrolls chat to message containing user input when down arrow button clicked
document.getElementById('search-next').addEventListener('click', () => {
    let search = document.getElementById('search');
    // ignore input < 3 in length
    if (search.value.length < 3) {
        search_error_msg.innerText = 'Please enter word length > 2';
        search_error_msg.style.display = 'block';
        return;
    }
    search_error_msg.style.display = 'none';

    let sentences = Array.from(document.getElementsByClassName('sentences'));

    // reset search when there is new input typed
    if (prevSearch !== search.value) {
        index = 0;
        prevSearch = search.value;
    }

    index = (index === sentences.length) ? 0 : index;
    const oldIndex = index;

    // find and scroll to the message containing the user input 
    for (let i = index; i < sentences.length; ++i) {
        if (sentences[i].innerText.toLowerCase().indexOf(search.value.toLowerCase()) !== -1) {
            sentences[i].scrollIntoView({ behavior: 'smooth' });
            index = i;
            break;
        }
    }

    if (index === 0 || oldIndex === index) {
        search_error_msg.innerText = 'No more results!';
        search_error_msg.style.display = 'block';
    }
    if (oldIndex !== index) {
        ++index;
    }
});

// show 'viewing older messages' when user scrolls chat past the last question/answer
document.getElementById('chatBox').addEventListener('scroll', () => {
    var div = document.getElementById('chatBox');

    const appended_sentences = document.getElementById('appended_sentences');
    const chat_window = document.getElementById('chatbot_window');
    const chat_height = div.scrollHeight - div.clientHeight;
    let last_element_height = 0;

    // get height of last message div in chat
    if (appended_sentences && appended_sentences.hasChildNodes()) {
        last_element_height = appended_sentences.lastChild.clientHeight;

    } else if (chat_window.hasChildNodes()) {
        last_element_height = chat_window.lastChild.previousSibling.clientHeight;
    }

    // show 'viewing older messages' when scrolled past last message div
    let to_bottom = document.getElementById('to-bottom');
    if (chat_height - last_element_height > div.scrollTop) {
        to_bottom.className = to_bottom.className.replace(/hidden/g, '');
        
    } else {
        to_bottom.className += ' hidden';
    }
})



// scrolls chat to the bottom when arrow down button from 'viewing older messages' is clicked 
document.getElementById('go-to-bottom').addEventListener('click', () => {
    var div = document.getElementById('chatBox');
    div.scrollTop = div.scrollHeight - div.clientHeight;
})