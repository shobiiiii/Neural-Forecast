// Define the URL of the server-side endpoint
const url = '/backtest';

// Get references to the HTML elements that contain the user input
const tokenInput = document.getElementById('token');
const strategyInput = document.getElementById('strategy');
const startDateInput = document.getElementById('start-date');
const endDateInput = document.getElementById('end-date');

// Define a function to handle the form submission
function handleSubmit(event) {
  event.preventDefault(); // Prevent the default form submission behavior

  // Get the user input from the HTML elements
  const token = tokenInput.value;
  const strategy = strategyInput.value;
  const startDate = startDateInput.value;
  const endDate = endDateInput.value;

  // Define the data to send to the server
  const data = {
    token,
    strategy,
    startDate,
    endDate
  };

  // Define the options for the fetch request
  const options = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  };

  // Send the fetch request to the server
  fetch(url, options)
    .then(response => response.json())
    .then(data => {
      console.log(data);
      // Handle the response from the server here
    })
    .catch(error => {
      console.error(error);
      // Handle errors here
    });
}

// Add an event listener to the form submit button
const form = document.getElementById('submit');
form.addEventListener('click', handleSubmit);
