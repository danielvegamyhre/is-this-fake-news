const getPrediction = async (title, text) => {
    const response = await fetch('/predict', {
      method: 'POST',
      body: JSON.stringify({'title': title, 'text': text}),
      headers: {
        'Content-Type': 'application/json'
      }
    });
    const myJson = await response.json(); //extract JSON from the http response
    return myJson.result;
  }