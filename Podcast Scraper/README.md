# Dynamic Podcasts Episodes Scraper

A robust Python-based web scraping tool designed to extract podcast episode information from Apple Podcasts and Spotify. The script creates a comprehensive HTML output featuring episode details with direct links to both platforms.

## Features

- **Multi-Platform Scraping**: Extracts episode information from both Apple Podcasts and Spotify
- **Dynamic Loading**: Handles dynamically loaded content through automated scrolling
- **Comprehensive Data Extraction**: Captures episode titles, descriptions, and platform-specific URLs
- **Responsive HTML Output**: Generates a clean, mobile-friendly HTML page with episode cards
- **Platform-Specific Links**: Provides direct links to episodes on both Apple Podcasts and Spotify
- **Robust Error Handling**: Implements retry mechanisms for improved reliability

## Prerequisites

Before running the script, ensure you have the following installed:

```bash
pip install selenium
pip install chromedriver-autoinstaller
pip install webdriver-manager
```

## System Requirements

- Python 3.x
- Chrome Browser
- Adequate internet connection for web scraping

## Installation

1. **Clone the Repository**:
```bash
git clone https://github.com/your-username/python-projects.git
cd python-projects
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

1. **Configure Chrome Settings**:
```python
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
```

2. **Run the Scraper**:
```python
url = "https://podcasts.apple.com/za/podcast/your-podcast-url"
apple_episodes = scrape_episodes(url)

url = "https://open.spotify.com/show/your-show-id"
spotify_episodes = scrape_episodes(url)
```

3. **Generate HTML Output**:
```python
episodes = create_episode_info(apple_episodes, spotify_episode_urls)
print_formatted_episodes_html(episodes)
```

## Key Components

- `init_driver()`: Initializes the Chrome WebDriver with appropriate settings
- `scroll_to_bottom()`: Handles dynamic content loading through automated scrolling
- `extract_episode_info()`: Extracts detailed information from episode elements
- `scrape_episodes()`: Main function coordinating the scraping process
- `create_episode_info()`: Combines data from both platforms
- `print_formatted_episodes_html()`: Generates responsive HTML output

## Output Format

The script generates an HTML file with:
- Responsive episode cards
- Platform-specific links (Apple Podcasts & Spotify)
- Episode titles and descriptions
- Mobile-friendly layout
- Clean, modern styling

## Error Handling

The script includes:
- Multiple retry attempts for failed elements
- Explicit waits for dynamic content
- Exception handling for common scraping issues
- Detailed error logging

## Future Improvements

- **YouTube URL Integration**: Add functionality to extract and display corresponding YouTube URLs (if available) for each episode
- **Improved Ordering Mechanism**: Expand the order toggle feature to sort episodes based on criteria like popularity or duration
- **Enhanced Error Handling**: Implement more sophisticated retry mechanisms and detailed error logs
- **Multithreading Support**: Allow for parallel scraping of multiple podcast pages
- **Data Storage Options**: Add support for saving scraped data to CSV, JSON, or database formats

## Limitations

- Requires stable internet connection
- Subject to platform-specific rate limiting
- Dependent on platform HTML structure
- May need updates if platforms change their layouts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Roscoe Kerby - ROSCODE**
- GitHub: [roscoekerby](https://github.com/roscoekerby)
- LinkedIn: [Roscoe Kerby](https://www.linkedin.com/in/roscoekerby/)

## Acknowledgments

- Selenium WebDriver for providing web automation capabilities
- Chrome WebDriver for browser automation support
- All contributors who help improve this project
