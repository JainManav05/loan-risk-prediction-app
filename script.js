document.addEventListener('DOMContentLoaded', () => {
    // Make sliders interactive by updating their output value
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const output = document.querySelector(`output[for="${slider.id}"]`);
        if (output) {
            output.textContent = slider.value;
            slider.addEventListener('input', () => {
                output.textContent = slider.value;
            });
        }
    });

    // Get all necessary DOM elements
    const form = document.getElementById('loan-form');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('result-container');
    const explanationBtn = document.getElementById('show-explanation-btn');
    const explanationContainer = document.getElementById('explanation-container');
    
    let currentExplanation = null;

    // Handle the main form submission
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        
        loader.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        explanationContainer.classList.add('hidden');
        explanationBtn.classList.add('hidden');

        const loanData = {
            loan_amnt: parseFloat(document.getElementById('loan_amnt').value),
            int_rate: parseFloat(document.getElementById('int_rate').value),
            installment: parseFloat(document.getElementById('installment').value),
            annual_inc: parseFloat(document.getElementById('annual_inc').value),
            dti: parseFloat(document.getElementById('dti').value),
            grade: document.getElementById('grade').value,
            emp_length: document.getElementById('emp_length').value,
            home_ownership: document.getElementById('home_ownership').value,
            verification_status: document.getElementById('verification_status').value,
            purpose: document.getElementById('purpose').value,
            title: document.getElementById('title').value
        };

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(loanData),
        })
        .then(response => {
            if (!response.ok) { throw new Error('Network response was not ok'); }
            return response.json();
        })
        .then(data => {
            loader.classList.add('hidden');
            resultContainer.classList.remove('hidden');
            
            updateGauge(data.default_probability);

            currentExplanation = data.explanation;
            explanationBtn.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            loader.classList.add('hidden');
            alert('Error: Could not get a prediction. Please ensure the API server is running.');
        });
    });

    // Add click listener for the "Show Explanation" button
    explanationBtn.addEventListener('click', () => {
        if (currentExplanation) {
            displayExplanation(currentExplanation);
            explanationContainer.classList.toggle('hidden');
        }
    });

    // --- ALL FUNCTIONS MOVED INSIDE DOMContentLoaded ---

    function updateGauge(probability) {
        const percentage = probability * 100;
        const gaugeFill = document.querySelector('.gauge__fill');
        const gaugeText = document.querySelector('.gauge__text');
        const riskLevelText = document.querySelector('.risk-level');

        if (gaugeFill && gaugeText && riskLevelText) {
            const rotation = (percentage / 100) * 180;
            gaugeFill.style.transform = `rotate(${rotation}deg)`;
            gaugeText.textContent = `${percentage.toFixed(2)}%`;
            
            let riskLevel, color;
            if (percentage < 20) {
                riskLevel = 'Low Risk';
                color = 'var(--low-risk-color)';
            } else if (percentage < 50) {
                riskLevel = 'Medium Risk';
                color = 'var(--medium-risk-color)';
            } else {
                riskLevel = 'High Risk';
                color = 'var(--high-risk-color)';
            }
            
            gaugeFill.style.background = color;
            riskLevelText.textContent = riskLevel;
            riskLevelText.style.color = color;
        }
    }
function displayExplanation(explanation) {
    const chart = document.getElementById('explanation-chart');
    if (!chart) return;

    chart.innerHTML = ''; // Clear previous chart

    // --- DICTIONARY TO TRANSLATE FEATURE NAMES ---
    const featureDescriptions = {
        'loan_amnt': 'A high loan amount',
        'annual_inc': 'A high annual income',
        'int_rate': 'A high interest rate',
        'dti': 'A high debt-to-income ratio',
        'grade_A': 'An excellent loan grade (A)',
        'grade_B': 'A good loan grade (B)',
        'grade_C': 'An average loan grade (C)',
        'grade_D': 'A below-average loan grade (D)',
        'emp_length_10+ years': 'A long employment history',
        'home_ownership_MORTGAGE': 'Having a mortgage'
        // Add more translations as needed
    };

    explanation.forEach(item => {
        // Find a user-friendly description, or use the raw feature name as a fallback
        const description = featureDescriptions[item.feature] || item.feature.replace(/_/g, ' ');

        let sentence = '';
        // Check if the SHAP value is positive (increases risk) or negative (decreases risk)
        if (item.value > 0) {
            sentence = `<strong>${description}</strong> increased the predicted risk.`;
        } else {
            sentence = `<strong>${description}</strong> decreased the predicted risk.`;
        }
        
        // Create a paragraph element for the sentence
        const p = document.createElement('p');
        p.innerHTML = sentence; // Use innerHTML to render the <strong> tag
        p.className = 'explanation-sentence'; // Add a class for styling
        
        chart.appendChild(p);
    });
}
}); // <-- End of the DOMContentLoaded listener