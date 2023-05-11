// Define the URL of the server-side endpoint
const url = 'http://127.0.0.1:5000/submit';

// Get references to the HTML elements that contain the user input
const assetInput = document.getElementById('asset');
const strategyInput = document.getElementById('strategy');
const startDateInput = document.getElementById('start-date');
const endDateInput = document.getElementById('end-date');

// Define a function to handle the form submission
function handleSubmit(event) {
  event.preventDefault(); // Prevent the default form submission behavior

  // Get the user input from the HTML elements
  const asset = assetInput.value;
  const assetType = assetInput.options[assetInput.selectedIndex].parentNode.label;
  const strategy = strategyInput.value;
  const startDate = startDateInput.value;
  const endDate = endDateInput.value;

  // Define the data to send to the server
  const data = {
    asset,
    assetType,
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
      // Update the src attribute of the img tag to display the plot
      document.getElementById('plotImage').src = data.image_url;
      // Show the div
      document.getElementById('plot').style.display = 'block';
    })
    .catch(error => {
      console.error(error);
      // Handle errors here

    });
}

// Add an event listener to the form submit button
const form = document.getElementById('submit');
form.addEventListener('click', handleSubmit);
