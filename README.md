# Route Optimization

## Introduction
This Route Optimization Application is a Streamlit-based web tool designed to efficiently manage and optimize delivery routes. It is built with a focus on ease of use, integrating various functionalities like distance matrix calculations, data encryption, and route plotting, and transferring of data using the OCEAN protocol.

## Getting Started
### Prerequisites
- Python 3.8+
- Streamlit
- OCEAN.py

## Installation
1. Clone the repository to your local machine.
2. Install the required Python packages:
   ```python
   pip install -r requirements.txt
   ```
3. Ensure you have Streamlit installed. If not, install it using:
   ```python
   pip install streamlit
   ```

## Running the Application

To run the application, navigate to the project directory and execute the following command in the terminal:
```python
streamlit run app.py
```
Executing app.py should open the application in your default web browser. 
## Features

- **Data Loading:** Ability to load and process data from various sources.
- **Route Optimization:** Implements algorithms to optimize delivery routes for efficiency.
- **Distance Matrix Calculation:** Supports creating distance matrices using different APIs.
- **Data Encryption and Transfer:** Offers functionality to encrypt and transfer data securely.
- **Interactive UI:** A user-friendly interface to input data, view, and interact with the optimized routes.

## Data Files

The application uses several data files to operate:

- `vehicles_lmd.csv`: Information about the vehicles.
- `orders_lmd.csv`: Details of the orders to be delivered.
- `depots_lmd.csv`: Data regarding depots.
- `helper_funcs.py`: A module containing helper functions for various tasks.
