function make_text_area(name, placeholder, width, rows) {
    const text_area = document.createElement('textarea');
    text_area.className = 'form-control';
    text_area.name = name;
    text_area.placeholder = placeholder;
    text_area.style.borderRadius = '5px';
    text_area.style.width = width;
    text_area.rows = rows;
    text_area.style.resize = 'none';
    text_area.style.padding = '10px';
    text_area.style.margin = '0 auto';
    text_area.style.marginTop = '3%';
    return text_area;
}

function make_hr() {
    const hr = document.createElement('hr');
    hr.style.width = '80%';
    return hr;
}

function make_br() {
    const br = document.createElement('br');
    return br;
}

function make_option(value) {
    const option = document.createElement('option');
    option.textContent = value;
    option.value = value;
    return option;
}

function make_header(text, size = 'default') {
    const header = document.createElement('h1');
    header.textContent = text;
    header.style.textAlign = 'center';
    header.style.color = 'black';
    header.style.fontWeight = '800';
    header.style.fontFamily = 'monospace';
    header.style.fontSize = size;
    return header;
}

function make_star(value) {
    const star = document.createElement('button');
    star.className = 'star';
    star.name = 'star';
    star.type = 'button';
    star.innerHTML = '&#9733;';
    star.value = value;
    star.style.backgroundColor = 'transparent';
    star.style.border = 'none';
    star.style.outline = 'none';
    star.style.fontSize = '500%';
    star.style.color = '#303030';
    star.style.width = '20%';
    star.style.marginTop = '-4%';

    star.addEventListener('click', (e) => {
        const stars = document.getElementsByClassName('star');
        const rating = document.getElementsByName('user_rating')[0];
    
        Array.from(stars).map(s => {s.style.color = '#303030';});
        
        for (let i = 0; i < e.target.value; ++i) {
            stars[i].style.color = '#33ccff';             
        }
        rating.value = e.target.value;
    });
    return star;
}

function make_hidden(name, value) {
    const element = document.createElement('textarea');
    element.name = name;
    element.value = value;
    element.style.display = 'none';
    return element;
}

function make_input(type, name, placeholder) {
    const input = document.createElement('input');
    input.type = type;
    input.name = name;
    input.className = 'form-control';
    input.placeholder = placeholder;
    input.style.width = '70%';
    input.style.margin = '0 auto';
    input.style.marginBottom = '2%';
    input.required = true;
    return input;
}


// submit feedback input
const feedback_body = document.getElementById('Feedback');
feedback_body.style.height = '600px';
const submit_button = document.getElementById('feedback_button');
submit_button.addEventListener('click', () => {
    let feedback = document.getElementsByName('user_feedback')[0]
    let category = document.getElementsByName('selected_category')[0]
    let rating = document.getElementsByName('user_rating')[0]

    fetch('/feedback', {
        method: 'post',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user_feedback: feedback.value, selected_category: category.value, user_rating: rating.value})
    })
    .then(response => {return response.text();})
    .then(text => {
        console.log("Feedback submit " + text);

        // reset form
        const stars = document.getElementsByClassName('star');
        Array.from(stars).map(s => { s.style.color = '#303030'; });
        rating.value = 0;
        category.value = 'Suggestions'
        feedback.value = ''
    });
    
    window.alert("\nThank you for your feedback!");
});

// feedback category dropdown
const category_dropdown = document.createElement('select');
category_dropdown.className = 'custom-select';
category_dropdown.name = 'selected_category';
category_dropdown.style.width = '70%';
category_dropdown.style.display = 'block';
category_dropdown.style.marginLeft = '25%';
category_dropdown.style.margin = '0 auto';
category_dropdown.style.marginTop = '3%';
category_dropdown.appendChild(make_option('Suggestions'));
category_dropdown.appendChild(make_option('Issues'));
category_dropdown.appendChild(make_option('General'));

// feedback star rating
const star_div = document.createElement('div');
star_div.style.margin = '0 auto';
star_div.style.width = '50%';
star_div.appendChild(make_star(1));
star_div.appendChild(make_star(2));
star_div.appendChild(make_star(3));
star_div.appendChild(make_star(4));
star_div.appendChild(make_star(5));
star_div.appendChild(make_hidden('user_rating', 0));

// feedback contents
feedback_body.prepend(make_text_area('user_feedback', 'Write your feedback here ...', '70%', '5'));
feedback_body.prepend(make_header('Leave your feedback below'));
feedback_body.prepend(make_hr());
feedback_body.prepend(make_br());
feedback_body.prepend(category_dropdown);
feedback_body.prepend(make_header('Feedback category'));
feedback_body.prepend(make_hr());
feedback_body.prepend(star_div);
feedback_body.prepend(make_header('Rate your experience'));
feedback_body.prepend(make_br());


// submit contact us input
const contact_body = document.getElementById('ContactUs');
contact_body.style.height = '600px';
const contact_submit = document.getElementById('contact_us_button')
contact_submit.addEventListener('click', () => {
    let name = document.getElementsByName('contact_name')[0];
    let email = document.getElementsByName('contact_email')[0];
    let message = document.getElementsByName('contact_message')[0];

    fetch('/contact-us', {
        method: 'post',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({contact_name: name.value, contact_email: email.value, contact_message: message.value})
    })
    .then(response => {return response.text();})
    .then(text => {
        console.log("Contact Submit " + text);

        // reset form
        name.value = ''
        email.value = ''
        message.value = ''
    });
   window.alert('Thank you for contacting us!');
});

// contact us header
const p = document.createElement('p');
p.textContent = 'Please leave any questions or inquiries below.';
p.style.textAlign = 'center';
p.style.color = 'black';
p.style.fontWeight = '300';
p.style.fontFamily = 'monospace';
p.style.fontSize = '20px';
p.style.marginBottom = '5%';

// contact us contents
contact_body.prepend(make_text_area('contact_message', 'Message', '70%', '10'));
contact_body.prepend(make_input('email', 'contact_email', 'Email'));
contact_body.prepend(make_input('text', 'contact_name', 'Name'));
contact_body.prepend(p);
contact_body.prepend(make_header('Contact Us', '50px'));
contact_body.prepend(make_br());


if (document.getElementById('stars-filled')) {
    document.getElementById('stars-filled').style.width = document.getElementById('star-value').innerText*20 + '%';
}