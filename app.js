// Define the URL of the server-side endpoint
const url = 'https://server.jasemjasem.ir/submit';

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
      console.log(data.image_url)
      console.log('Hello.')
      // Update the src attribute of the img tag to display the plot
      document.getElementById('plotImage').src = data.image_url;

      // Show the div
      document.getElementById('plot').style.display = 'block';

      // Assuming you have fetched the data from your server and stored it in variables
      const winRateText = data.win_rate;
      const rewardRiskText = data.reward_risk;
      const drawdownText = data.drawdown;

      // Show the div
      document.getElementById('statement').style.display = 'block';

      // Update the content of the headings
      document.querySelector("#statement h3:nth-of-type(1)").textContent += `: ${winRateText}`;
      document.querySelector("#statement h3:nth-of-type(2)").textContent += `: ${rewardRiskText}`;

      console.log('Done.')
    })
    .catch(error => {
      console.error(error);
      // Handle errors here
    });
}

// Add an event listener to the form submit button
const form = document.getElementById('submit');
form.addEventListener('click', handleSubmit);
