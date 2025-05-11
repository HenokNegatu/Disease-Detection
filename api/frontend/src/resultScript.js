const results = JSON.parse(localStorage.getItem('diagnosisResults'));

document.getElementById('diseaseName').textContent = results.disease;

const symptomsList = document.getElementById('symptomsList');
results.symptoms.forEach(symptom => {
    const li = document.createElement('li');
    li.textContent = symptom;
    symptomsList.appendChild(li);
});

const treatmentsList = document.getElementById('treatmentsList');
results.treatments.forEach(treatment => {
    const li = document.createElement('li');
    li.textContent = treatment;
    treatmentsList.appendChild(li);
});
