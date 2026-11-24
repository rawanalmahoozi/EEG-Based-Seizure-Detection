<h1 align="center">
  <img src="https://github.com/user-attachments/assets/a255c068-4893-45b5-9372-a58c97a630c7"
       width="55"
       align="center"
       alt="NeuroReview Logo">
  NeuroReview
</h1>

<p align="center">
  EEG Analysis & Specialist Review
</p>### EEG Analysis & Specialist Review

## About the Project

Understanding EEG recordings can be a time-consuming process, especially when working with long recordings that may contain seizure activity. NeuroReview was developed to explore how machine learning can support this process by helping identify potentially relevant segments and making them easier to review.

NeuroReview is a web-based application built with Python and Streamlit. It uses a trained Convolutional Neural Network (CNN) to analyse EEG recordings and highlight segments that may contain seizure activity. Rather than replacing specialist assessment, the application provides a structured way to explore model predictions, review EEG signals, and document findings.

The project brings automated analysis and specialist review together in one interface, from uploading an EEG recording to generating a report of the reviewed results.

> **Important:** NeuroReview is a research prototype for educational and research purposes. It is not intended for clinical diagnosis or treatment. All model predictions require specialist review.

## How It Works

The process begins when an EEG recording in EDF format is uploaded to the application. The recording is prepared for analysis, including selecting the required EEG channels and processing the signal into smaller segments.

Each segment is then passed through the trained CNN model, which evaluates the signal and produces a prediction. Segments identified as potentially containing seizure activity are highlighted for further inspection.

Once the analysis is complete, the results are presented in the application. Users can explore flagged segments, view their EEG signals, and examine details such as the segment's start time and duration.

This allows the analysis results to be reviewed in a more organized way, rather than requiring users to navigate the entire recording manually.

## Key Features

NeuroReview combines several features to support the analysis and review process:

- **EEG Recording Analysis:** Upload EDF recordings and process them using the trained CNN model.
- **Automated Seizure Detection:** Highlight segments identified by the model as potentially containing seizure activity.
- **EEG Signal Visualization:** Display EEG signals to help users inspect flagged segments.
- **Specialist Review:** Record whether a flagged segment is a confirmed seizure, not a seizure, or requires follow-up.
- **Review Notes:** Add notes and save decisions for individual segments.
- **PDF Reporting:** Generate reports summarizing the analysis and specialist review results.

## Dataset

The model was developed using the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/), available through PhysioNet.

The database contains scalp EEG recordings in EDF format, along with annotations identifying seizure activity. These recordings provide the data used to prepare and train the model to distinguish between normal and seizure-related EEG segments.

## Model & Approach

NeuroReview uses a one-dimensional Convolutional Neural Network (1D CNN), designed to learn patterns directly from EEG signals.

For model preparation, the recordings are processed using 18 bipolar EEG channels and resampled to 128 Hz. They are then divided into overlapping 8-second segments, allowing the model to analyse the recordings in smaller portions.

The CNN processes these segments and classifies them into two categories: normal and seizure. Its predictions are then used to identify segments that should be highlighted for further review.

Model performance is assessed using several evaluation metrics, including accuracy, precision, recall, and F1-score, providing a broader view of its classification performance.

## Application Preview

The application provides a central dashboard where users can access analysed recordings, explore flagged segments, and manage specialist reviews.

**Live Application:** https://eeg-seizure-detection-assistant.streamlit.app/

**Demo Video:** 




https://github.com/user-attachments/assets/5bd9e0b6-736f-4522-ad81-ed61deb6440c









## Getting Started

To run NeuroReview locally, first clone the repository


Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then launch the application:

```bash
streamlit run app.py
```

Ensure that the trained model and its corresponding normalization files are available in the project directory before starting the application.

## Project Structure

The application is organized into separate files to keep its functionality and presentation manageable:

```text
NeuroReview/
│
├── app.py                  # Main application
├── features.py             # Application features
├── style.py                # Interface styling
├── retrain.py              # Model retraining
├── requirements.txt        # Project dependencies
├── brain_icon.png          # Application icon
│
├── model_cnn_8s_final.keras
├── cnn_mean_8s.npy
└── cnn_std_8s.npy
```

The project also includes a retraining script that allows the model to be updated using specialist-reviewed segments. This provides a starting point for exploring how review feedback can contribute to future model improvements.

## Future Directions

Future development of NeuroReview could focus on incorporating additional patient information to provide more context when reviewing EEG recordings.

One important improvement would be to include **medication information**, such as whether a patient is taking their prescribed medication consistently. This could help specialists understand whether seizure activity occurs despite medication use or in situations where doses may have been missed.

Another valuable addition would be recording **potential seizure triggers**, such as sleep deprivation, stress, illness, or other factors that may be associated with seizure episodes. Having this information alongside EEG analysis could help specialists explore possible patterns and better understand the circumstances surrounding seizure activity.

Further improvements could also include expanding the dataset, evaluating model performance on additional recordings and incorporating more specialist feedback to support future model development.

Together, these additions could help NeuroReview move beyond identifying potentially relevant EEG segments towards providing a more informative and context-aware review process.

## Author

**Rawan Almahoozi**

[LinkedIn](https://www.linkedin.com/in/rawan-almahoozi-b0a82a364)<img width="280" height="241" alt="brain_icon" src="https://github.com/user-attachments/assets/f33ecf03-e6cd-4b03-b1a5-dca626dcd8f2" />
