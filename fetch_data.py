import pandas as pd
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

def scrape_topcoder_challenges(max_pages_to_load=40, target_count=300):
    print(f"Starting Topcoder scraper. Target: {target_count} projects.")
    
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--lang=en-US") 
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    challenge_urls = set()
    dataset = []
    csv_filename = "topcoder_dataset.csv"
    
    try:
        base_url = "https://www.topcoder.com/challenges?bucket=allPast&tracks[DS]=true&tracks[Des]=true&tracks[Dev]=true&tracks[QA]=true&types[]=CH&types[]=F2F&types[]=MM&types[]=TSK"
        driver.get(base_url)
        time.sleep(15) 
        
        last_link_count = 0
        
        # Phase 1: Collect challenge URLs via pagination
        for i in range(max_pages_to_load):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            try:
                xpath_expr = "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'view more') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'load more')]"
                view_more_btn = driver.find_element(By.XPATH, xpath_expr)
                driver.execute_script("arguments[0].click();", view_more_btn)
                time.sleep(5)
            except Exception:
                time.sleep(3) 
            
            current_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/challenges/')]")
            challenge_urls.clear()
            for elem in current_links:
                href = elem.get_attribute("href")
                if href and len(href.split('/challenges/')) > 1:
                    clean_href = href.split('?')[0] 
                    challenge_urls.add(clean_href)
                    
            print(f"Loaded page {i+1}/{max_pages_to_load} | Found {len(challenge_urls)}/{target_count} URLs")
            
            if len(challenge_urls) >= target_count:
                print(f"Target of {target_count} reached. Stopping pagination.")
                break
                
            if len(current_links) > 0 and len(current_links) == last_link_count:
                 print("Pagination exhausted. Moving to extraction.")
                 break
            last_link_count = len(current_links)

        challenge_urls = list(challenge_urls)[:target_count]
        print(f"\nFound {len(challenge_urls)} unique challenge URLs. Starting extraction...\n")

        # Phase 2: Extract data from each challenge page
        for idx, url in enumerate(challenge_urls, 1): 
            try:
                driver.get(url)
                time.sleep(8)
                
                # Extract Title
                try:
                    title = driver.find_element(By.XPATH, "//h1 | //h2").text.strip()
                except Exception:
                    title = "N/A"
                    
                # Extract Prize (Regex match)
                page_text = driver.find_element(By.TAG_NAME, "body").text
                prize = None
                prize_match = re.search(r'\$([\d.,]+)', page_text)
                if prize_match:
                    try:
                        raw_num = prize_match.group(1)
                        clean_num = re.sub(r'[.,]\d{2}$', '', raw_num)
                        clean_num = re.sub(r'[.,]', '', clean_num)
                        prize = float(clean_num)
                    except ValueError:
                        pass
                
                # Extract Duration (Heuristic match for days/weeks/months)
                duration_days = 7 # Default fallback
                duration_match = re.search(r'(\d+)\s*(day|days|week|weeks|month|months)', page_text, re.IGNORECASE)
                if duration_match:
                    try:
                        num = int(duration_match.group(1))
                        unit = duration_match.group(2).lower()
                        if 'week' in unit:
                            duration_days = num * 7
                        elif 'month' in unit:
                            duration_days = num * 30
                        else:
                            duration_days = num
                    except Exception:
                        pass
                        
                # Extract main description block
                description = "N/A"
                try:
                    main_content = driver.find_elements(By.XPATH, "//div[p or h3 or ul]")
                    largest_text = max([element.text for element in main_content], key=len, default="")
                    
                    if len(largest_text) > 100:
                        description = largest_text
                    else:
                        description = page_text.split("Challenge Summary")[-1] if "Challenge Summary" in page_text else page_text.split("Details")[-1]
                except Exception:
                    description = page_text
                    
                description = re.sub(r'\n+', ' | ', description).strip()[:4000]

                # Identify core technologies
                known_techs = ["Java", "Python", "React", "Node", "Angular", "Vue", "iOS", "Android", "C++", "C#", "AWS", "Design", "Figma", "UI", "QA", "Testing", "SQL", "Docker"]
                techs_found = [tech for tech in known_techs if tech.lower() in page_text.lower()]
                technologies = ", ".join(techs_found) if techs_found else "N/A"
                
                # Validate and append
                if title != "N/A" and prize is not None:
                    dataset.append({
                        "Title": title,
                        "URL": url,
                        "Technologies": technologies,
                        "Prize_USD": prize,
                        "Duration_Days": duration_days,
                        "Description": description
                    })
                    print(f"[{idx}/{len(challenge_urls)}] Extracted: {title[:30]}... (${prize} | {duration_days} days)")
                    
                    # --- AUTO-SAVE --- 
                    pd.DataFrame(dataset).to_csv(csv_filename, index=False, encoding='utf-8')
                    
            except Exception as e:
                print(f"Failed to process {url}: {e}")
                continue

    finally:
        # --- SAFE QUIT ---
        try:
            driver.quit()
        except Exception:
            pass

    if dataset:
        print(f"\nExtraction complete. Exported {len(dataset)} records to {csv_filename}")
    else:
        print("\nExtraction failed. No valid records found.")

if __name__ == "__main__":
    scrape_topcoder_challenges(max_pages_to_load=40, target_count=300)