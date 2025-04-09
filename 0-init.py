import requests
from bs4 import BeautifulSoup
from pprint import pprint
import pandas as pd

def test_table_extraction(url):
    """
    Test function to extract table headers and first column values without sections
    
    Args:
        url (str): Full URL of the product page to test
    Returns:
        dict: Table data as key-value pairs
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table
        table = soup.find('div', {'class': 'shoe-review'}).find('table', {'class': 'table table-bordered table-hover'})


        if not table:
            print("No table found on the page")
            return None

        # Initialize data storage
        table_data = {}

        # Process each row
        for row in table.find_all('tr'):
            # Skip section headers (th with colspan=3)
            if row.find('th', colspan='3'):
                continue

            # Get the header (th) and value (td) for regular rows
            header = row.find('th', style='font-weight:300;')
            value = row.find('td')
            
            if header and value:
                # Clean the header text (remove spans)
                header_text = header.text.strip().split('  ')[0]
                value_text = value.text.strip()
                
                # Store without section prefix
                table_data[header_text] = value_text

        return table_data

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    test_url = "https://runrepeat.com/nike-pegasus-41"
    result = test_table_extraction(test_url)
    
    if result:
        # Create a single row DataFrame
        transformed_data = {}
        
        # Process each metric and its value
        for metric, value in result.items():
            # Handle weight separately
            if metric == 'Weight':
                try:
                    oz = float(value.split(' oz')[0])
                    g = float(value.split('(')[1].split('g)')[0])
                    transformed_data['Weight (oz)'] = oz
                    transformed_data['Weight (g)'] = g
                    continue
                except:
                    transformed_data['Weight (oz)'] = None
                    transformed_data['Weight (g)'] = None
                    continue
            
            # Skip these columns - keep as is
            if metric in ['Size', 'Price', 'Reflective elements', 'Tongue: gusset type', 
                         'Heel tab', 'Removable insole']:
                transformed_data[metric] = value
                continue
            
            # Process numeric values with units
            try:
                if 'mm' in value:
                    transformed_data[f"{metric} (mm)"] = float(value.split(' mm')[0])
                elif 'HA' in value:
                    transformed_data[f"{metric} (HA)"] = float(value.split(' HA')[0])
                elif 'HC' in value:
                    transformed_data[f"{metric} (HC)"] = float(value.split(' HC')[0])
                elif '%' in value:
                    transformed_data[f"{metric} (%)"] = float(value.split('%')[0])
                elif 'N' in value:
                     transformed_data[f"{metric} (N)"] = float(value.split('N')[0])
                elif value.isdigit():
                    transformed_data[metric] = float(value)
                else:
                    transformed_data[metric] = value
            except:
                transformed_data[metric] = value

        # Create DataFrame with a single row
        df = pd.DataFrame([transformed_data])
        
        # Sort columns alphabetically
        df = df.reindex(sorted(df.columns), axis=1)
        
        print("\n=== Transformed Table Data ===")
        print(df)
        
        # Save to CSV
        df.to_csv('shoe_metrics_transformed.csv', index=False)
    else:
        print("No data extracted.")


def extract_review_date(soup):
    """
    Extract the review date from the author information section
    
    Args:
        soup (BeautifulSoup): BeautifulSoup object of the page
    Returns:
        tuple: (original_date, updated_date) or (original_date, None) if no update date
    """
    try:
        # Find the author-name div
        author_div = soup.find('div', class_='author-name')
        if not author_div:
            return None, None

        # Get the full text content
        text_content = author_div.text.strip()
        
        # Split by "on" and get the date part
        date_parts = text_content.split(' on ')[-1].strip()
        
        # Extract original date
        original_date = None
        updated_date = None
        
        if '- updated' in date_parts:
            # Split into original and updated dates
            original_date = date_parts.split('- updated')[0].strip()
            updated_date = date_parts.split('- updated')[1].strip()
        else:
            original_date = date_parts.strip()
            
        return original_date, updated_date
        
    except Exception as e:
        print(f"Error extracting review date: {str(e)}")
        return None, None

# Test the function
url = "https://runrepeat.com/nike-pegasus-41"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

original_date, updated_date = extract_review_date(soup)
print(f"Original Date: {original_date}, Updated Date: {updated_date}")