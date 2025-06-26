# Lowpass Filter Project

This project generates a lowpass filter frequency response with a cutoff at 1.5 GHz and plots the corresponding impulse response.

## Project Structure

```
lowpass-filter-project
├── src
│   ├── main.py        # Entry point of the application
│   └── utils.py       # Utility functions for filter creation and impulse response calculation
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:

```bash
python src/main.py
```

This will generate the lowpass filter frequency response and display the impulse response plot.

## Dependencies

The project requires the following Python packages:

- numpy
- scipy
- matplotlib

Make sure to have these installed in your Python environment.