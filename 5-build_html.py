import base64

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def read_binary_file(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_html():
    # Read the Python script
    python_code = read_file('4-app.py')
    
    # Read the CSV file
    csv_data = read_binary_file('FootwearData_cleaned.csv')
    
    # Read the HTML template
    html_template = read_file('research.html')
    
    # Replace placeholders
    html_content = html_template.replace('YOUR_PYTHON_CODE_HERE', python_code)
    html_content = html_content.replace('YOUR_CSV_DATA_HERE', csv_data)
    
    # Write the final HTML
    with open('research.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    create_html()