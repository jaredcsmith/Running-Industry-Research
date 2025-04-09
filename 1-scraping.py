# Libraries
# https://realpython.com/beautiful-soup-web-scraper-python/
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import re
from urllib.parse import urlparse, urljoin
import time
from datetime import timedelta

# This code is in 3 parts 
# 1. URL Scraping
# 2. Functions : To parse Product Name, Pros and Cons Wrapper.
# 3. Pros and Cons and Lab Test Results Scraping
# URL Definition
# url = 'https://runrepeat.com/on-cloudeclipse' ### Will iterate through the on-cloudeclipse to find all possible options
# response = requests.get(url)
# html_content = response.content
# soup = BeautifulSoup(html_content, 'html.parser')
# table = soup.find('table', {'class': 'table table-bordered table-hover'})

# (soup.prettify()) # Print the entire HTML content of the page
# print(soup.prettify()) # Print the entire HTML content of the page

# # Function to extract product name from URL
# def extract_product_name(url):
#    parsed_url = urlparse(url)
#    path = parsed_url.path.strip('/')
#    # Use regex to cleanly extract the product name
#    product_name = re.sub(r'[^a-zA-Z0-9\s]', '', path).replace('-', ' ').title()
#    return product_name

# x = extract_product_name(url)
# print(x)
# # Function to extract pros and cons 
# def extract_pros_and_cons(soup):
#    pros = []
#    cons = []
#    wrapper = soup.find('div', class_='good-bad-wrapper')
#    if wrapper:
#        # Extract pros
#        good_section = wrapper.find('div', {'id': 'the_good', 'class': 'good-bad gb-type-good'})
#        if good_section:
#            ul = good_section.find('ul')
#            if ul:
#                pros = [li.text.strip() for li in ul.find_all('li')]

#         # Extract cons
#        bad_section = wrapper.find('div', {'id': 'the_bad', 'class': 'good-bad gb-type-bad'})
#        if bad_section:
#            ul = bad_section.find('ul')
#            if ul:
#                cons = [li.text.strip() for li in ul.find_all('li')]
    
#    return pros, cons

# # Extract pros and cons
# pros, cons = extract_pros_and_cons(soup)
# print('Pros:', pros)
# print('Cons:', cons)

# # Function to find all table
# def extract_table(soup):
#     lab_test_section = soup.find('h2', {'id': 'lab-tests'})
#     if lab_test_section:
#         table = lab_test_section.find_next('table', {'class': 'table table-bordered table-hover'})
#         if table:
#             # Extract headers from <th>
#             headers = [th.text.strip() for th in table.find_all('th')]
            
#             # Extract only the first column of data
#             first_column_data = []
#             valid_headers = []
#             for i, row in enumerate(table.find_all('tr')):
#                 cells = row.find_all('td')
#                 if cells:
#                     first_column_data.append(cells[0].text.strip())
#                     if cells[0].text.strip():  # Check if the first column has a value
#                         valid_headers.append(headers[i])
            
#             return valid_headers, first_column_data
#     return None, None

# headers, first_column_data = extract_table(soup)
# print("Headers:", headers)
# print("First Column Data:", first_column_data)
# # check the length of headers and first_column_data
# print("Length of Headers:", len(headers))
# print("Length of First Column Data:", len(first_column_data))
# # Convert headers to a pandas Series and display the first few elements
# headers_series = pd.Series(headers)
# print(headers_series.tail())

# # Convert first_column_data to a pandas Series and display the first few elements
# first_column_data_series = pd.Series(first_column_data)
# print(first_column_data_series.tail())

# # Combine into a df
# df = pd.DataFrame({'Headers': headers, 'First Column Data': first_column_data})
# print(df.head())



# ### To scrape table of all information.
# def extract_table(soup, url, pros, cons):
#     lab_test_section = soup.find('h2', {'id': 'lab-tests'})
#     if lab_test_section:
#         table = lab_test_section.find_next('table', {'class': 'table table-bordered table-hover'})
#         if table:
#             # Classic data manipulation
#             matrix = []
#             for row in table.find_all('tr'):
#                 cells = [cell.text.strip() for cell in row.find_all('td')]
#                 if cells:
#                     matrix.append(cells)

#             # Remove the EXTRA column
#             filtered_matrix = [row[:2] for row in matrix if len(row) >= 2]

#             headers = [row[0] for row in filtered_matrix]
#             data = [row[1] for row in filtered_matrix]
#             product_name = extract_product_name(url)
#             final_matrix = []
#             header_row = ['Product Name'] + headers + ['Pros', 'Cons']  # HEADERS of columns
#             final_matrix.append(header_row)
#             data_row = [product_name] + data + [', '.join(pros), ', '.join(cons)]
#             final_matrix.append(data_row)

#             # Write the final matrix to the CSV file
#             csv_file = 'FootwearReviewDatabase.csv'

#             with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
#                 writer = csv.writer(file)
#                 writer.writerows(final_matrix)

#             print(f'Data has been written to {csv_file}')
#             return headers, data
#         else:
#             print('Table not found')
#             return None, None
#     return None, None

# # Extract pros and cons
# pros, cons = extract_pros_and_cons(soup)

# # Extract table with pros and cons
# headers, first_column_data = extract_table(soup, url, pros, cons)
# print("Headers:", headers)
# print("First Column Data:", first_column_data)


## Next steps
# https://runrepeat.com/catalog/running-shoes
# https://runrepeat.com/catalog/training-shoes
 #https://runrepeat.com/catalog/walking-shoes
# https://runrepeat.com/catalog/tennis-shoes
# https://runrepeat.com/catalog/sneakers
# https://runrepeat.com/catalog/track-spikes
# https://runrepeat.com/catalog/cross-country-shoes
# https://runrepeat.com/catalog/hiking-boots
# https://runrepeat.com/catalog/hiking-shoes
# https://runrepeat.com/catalog/hiking-sandals
# https://runrepeat.com/catalog/basketball-shoes
# https://runrepeat.com/catalog/training-shoes

# URLS 
#web_url = "https://runrepeat.com/catalog/"
#prod_url = "https://runrepeat.com/"
#product_categories = ["running-shoes", "training-shoes", "walking-shoes", "tennis-shoes", "sneakers","track-spikes", "cross-country-shoes","hiking-boots", "hiking-shoes", "hiking-sandals", "basketball-shoes", "training-shoes"]
### clicking into product name from catalog (Then perform the previous scraping code)
# <div class="product-name hidden-sm hidden-xs" data-v-a5a5736a="">
# <a href="/hoka-mach-5" target="_blank" data-v-a5a5736a="">
# <span data-v-a5a5736a="">Hoka Mach 5</span>
# </a> <!----></div>

# Click next page to scrape the next page of data 
# <a href="https://runrepeat.com/catalog/running-shoes?page=2" class="paginate-buttons next-button" disabled="false">›</a>
# ?page=2 append this until no more data remains


### Extra Variables 
## Awards and Verdict 
#<section id="product-intro" data-section-visiblity="1"><h2 class="product-intro-verdict">Our verdict</h2> <div class="">The Mach 5 from <a href="https://runrepeat.com/catalog/hoka-running-shoes">Hoka</a> is a performance trainer that can do a bit of everything. If you want to train with speed, we are convinced that it can become your trusty <a href="https://runrepeat.com/catalog/speed-training-running-shoes">tempo running shoe</a>. And if you want an easy-day or long-distance trainer, we believe that the Mach 5's cushy yet stable ride will serve you just right. Even better, it'll give you all the snappiness and energy return you need.</div> <ul class="awards-list" data-v-5ed0f565=""><!--[--><li class="awards-list__item" data-v-5ed0f565=""><span class="rank-text" data-v-5ed0f565="">Our top pick in </span> <a href="/guides/best-for-beginners-running-shoes" data-v-5ed0f565="">best Reviews of running shoes for beginners</a></li><!--]--></ul></section>
def scrape_data():
    start_time = time.time()
    ## Restructure Code Aug 27
    # Function to extract product name from URL
    def extract_product_name(url):
        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        product_name = path.replace('-', ' ')
        product_name = re.sub(r'[^a-zA-Z0-9\s]', '', product_name).title()
        product_name = product_name.replace(' Shoes', '')
        return product_name

    # Function to extract pros and cons
    def extract_pros_and_cons(soup):
        pros = []
        cons = []
        wrapper = soup.find('div', class_='good-bad-wrapper')
        if wrapper:
            # Extract pros
            good_section = wrapper.find('div', {'id': 'the_good', 'class': 'good-bad gb-type-good'})
            if good_section:
                ul = good_section.find('ul')
                if ul:
                    pros = [li.text.strip() for li in ul.find_all('li')]
            # Extract cons
            bad_section = wrapper.find('div', {'id': 'the_bad', 'class': 'good-bad gb-type-bad'})
            if bad_section:
                ul = bad_section.find('ul')
                if ul:
                    cons = [li.text.strip() for li in ul.find_all('li')] 
        return pros, cons

    # Function to extract Audience Rating
    def extract_audience_rating(soup):
        rating_div = soup.find('div', class_='corescore-big__score')
        if rating_div:
            return rating_div.text.strip()
        return None
    
    # Extract Verdict Data
# Extract Verdict Data
    def extract_verdict(soup):  
        verdict_title = soup.find('h2', class_='product-intro-verdict')
        verdict_text_div = verdict_title.find_next_sibling('div') if verdict_title else None
    
        if verdict_title and verdict_text_div:
        # Combine the title and text into a single string
            return f"{verdict_title.text.strip()}: {verdict_text_div.text.strip()}"
    
        return None

     # Function to extract additional features
    def extract_features(soup):
        features = {}
        # Extract Terrain value
        terrain = soup.find('span', class_='terrain-value')
        if terrain:
            spans = terrain.find_all('span')  # Find all <span> elements within terrain-value
            features['Terrain'] = ', '.join(span.text.strip() for span in spans if span.text.strip())
        else:
            features['Terrain'] = None

        # Extract Type value
        type = soup.find('span', class_='type-value')
        if type :
            spans = type .find_all('span')
            features['Type'] = ', '.join(span.text.strip() for span in spans if span.text.strip())
        else:
            features['Type'] = None

        
        # Extract toebox value
        toebox = soup.find('span', class_='toebox-value')
        if toebox:
            inner_span = toebox.find('span')  # Find the inner <span> that contains the text
            features['Toebox'] = inner_span.text.strip() if inner_span else None
        else:
            features['Toebox'] = None
        
        # Extract pace value
        pace = soup.find('span', class_='pace-value')
        if pace:
            inner_span = pace.find('span')  # Find the inner <span> that contains the text
            features['Pace'] = inner_span.text.strip() if inner_span else None
        else:
            features['Pace'] = None
        
        # Extract strike pattern value
        strike_pattern = soup.find('span', class_='strike-pattern-value')
        if strike_pattern:
            inner_span = strike_pattern.find('span')  # Find the inner <span> that contains the text
            features['Strike Pattern'] = inner_span.text.strip() if inner_span else None
        else:
            features['Strike Pattern'] = None
        
        # Extract material value
        material = soup.find('span', class_='material-value')
        if material:
            spans = material.find_all('span')  # Find all <span> elements within material-value
            unique_materials = set(span.text.strip() for span in spans if span.text.strip())
            features['Material'] = ', '.join(unique_materials)
        else:
            features['Material'] = None
        # Extract Release Date
        release_date = soup.find('span', class_='release-date-value')
        if release_date:
            inner_span = release_date.find('span')  # Find the inner <span> that contains the text
            features['Release Date'] = inner_span.text.strip() if inner_span else None
        else:
            features['Release Date'] = None
        return features
    
    # Function to extract table data
    def transform_table_data(table_data):
        """
        Transform raw table data into properly formatted columns with units
        
        Args:
            table_data (dict): Dictionary of raw table data
        Returns:
            dict: Transformed data with proper column names and units
        """
        transformed_data = {}
        
        for metric, value in table_data.items():
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
                
        return transformed_data
    
    def extract_review_date(soup):
        """
        Extract the review date from the author information section
        """
        try:
            author_div = soup.find('div', class_='author-name')
            if not author_div:
                return None, None

            text_content = author_div.text.strip()
            date_parts = text_content.split(' on ')[-1].strip()
            
            original_date = None
            updated_date = None
            
            if '- updated' in date_parts:
                original_date = date_parts.split('- updated')[0].strip()
                updated_date = date_parts.split('- updated')[1].strip()
            else:
                original_date = date_parts.strip()
                
            return original_date, updated_date
            
        except Exception as e:
            print(f"Error extracting review date: {str(e)}")
            return None, None
        
 # Replace the existing table extraction code in scrape_product function with:
    def scrape_product(url, category):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract pros and cons
        pros, cons = extract_pros_and_cons(soup)
        
        # Extract Audience Rating
        audience_rating = extract_audience_rating(soup)
        
        # Extract Our Verdict
        verdict = extract_verdict(soup)
        
        # Extract additional features
        features = extract_features(soup)
        original_date, updated_date = extract_review_date(soup)

        # Extract table data
        table = soup.find('table', {'class': 'table table-bordered table-hover'})
        if table:
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

            # Transform the table data to handle units
            transformed_table_data = transform_table_data(table_data)
            
            product_name = extract_product_name(url)
            
            # Consolidate all data
            final_data = {
                'Product Name': product_name,
                'URL': url,
                'Release Date': features['Release Date'],
                'Original Review Date': original_date,
                'Category': category,
                'Audience Rating': audience_rating,
                'RR Verdict': verdict,
                'Toebox': features['Toebox'],
                'Pace': features['Pace'],
                'Strike Pattern': features['Strike Pattern'],
                'Material': features['Material'],
                "Terrain": features['Terrain'],
                'Type': features['Type'],
                'Pros': ', '.join(pros),
                'Cons': ', '.join(cons),
                **transformed_table_data,  # Include transformed table data
            }
            return final_data
        else:
            return None

    # Function to extract product URLs from a catalog page
    def extract_product_urls(catalog_url):
        product_urls = []
        response = requests.get(catalog_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        product_elements = soup.find_all('div', class_='product-name')
        for element in product_elements:
            a_tag = element.find('a')
            if a_tag:
                product_url = urljoin(prod_url, a_tag['href'])
                product_urls.append(product_url)
        
        return product_urls

    # Function to handle pagination and collect all product URLs
    def get_all_product_urls(base_catalog_url):
        all_product_urls = []
        page = 1
        while True:
            catalog_url = f"{base_catalog_url}?page={page}"
            product_urls = extract_product_urls(catalog_url)
            if not product_urls:
                break
            all_product_urls.extend(product_urls)
            print(f"Collected {len(product_urls)} product URLs from page {page}.")
            page += 1
        print(f"Total product URLs collected: {len(all_product_urls)}")
        return all_product_urls

    # Category mapping
    category_mapping = {
        "running-shoes": "Running",
        "training-shoes": "Training",
        "walking-shoes": "Walking",
        "tennis-shoes": "Tennis",
        "sneakers": "Sneakers",
        "track-spikes": "Track",
        "cross-country-shoes": "Cross Country",
        "hiking-boots": "Hiking Boots",
        "hiking-shoes": "Hiking Shoes",
        "hiking-sandals": "Hiking",
        "basketball-shoes": "Basketball"
    }

    # Main function to scrape all products in a category
    def scrape_category(category):
        print(f"Scraping category: {category}")
        base_catalog_url = f"{web_url}{category}"
        all_product_urls = get_all_product_urls(base_catalog_url)
        
        all_data = []
        seen_product_names = set()
        for product_url in all_product_urls:
            product_data = scrape_product(product_url, category)
            if product_data and product_data['Product Name'] not in seen_product_names:
                # Map category to revised names
                product_data['Category'] = category_mapping.get(category, category).replace(' Shoes', '')
                all_data.append(product_data)
                seen_product_names.add(product_data['Product Name'])
        
        print(f"Collected data for category: {category} - {len(all_data)} products.")
        return all_data

    # Define URLs and categories
    web_url = "https://runrepeat.com/catalog/"
    prod_url = "https://runrepeat.com/"
    product_categories = [
        "running-shoes", "training-shoes", "walking-shoes", "tennis-shoes", 
        "sneakers", "track-spikes", "cross-country-shoes", "hiking-boots", 
        "hiking-shoes", "hiking-sandals", "basketball-shoes"
    ]

    # Scrape all categories and consolidate data into a single DataFrame
    all_data = []
    for category in product_categories:
        category_data = scrape_category(category)
        all_data.extend(category_data)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_data)
    df['Product Name'] = df['Product Name'].astype(str)
    # Function to capture the full brand name
    def capture_full_name(product_name):
        words = product_name.split()
        if words[0] == "Air" or words[0].lower().startswith("nikecourt"):
            return "Nike", ' '.join(words)
        elif words[0] in ["New", "Under", "K", "La"] and len(words) > 1:
            return words[0] + " " + words[1], ' '.join(words[2:])
        else:
            return words[0], ' '.join(words[1:])


    # Apply the function to split 'Product Name' into 'Brand' and 'Product Name'
    df[['Brand', 'Product Name']] = df['Product Name'].apply(lambda x: pd.Series(capture_full_name(x)))

    # Reorder columns if necessary
    col = df.pop('Brand')
    df.insert(0, 'Brand', col)

    df
    # Write the final consolidated data to a CSV file
    df.to_csv('FootwearData.csv', index=False)
    print('Data has been written to FootwearData.csv')
    end_time = time.time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"Elapsed time: {elapsed_time}")
    return df

scrape_data()