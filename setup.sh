mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
[theme]\n\
primaryColor = '#7792E3'\n\
backgroundColor = '#273346'\n\
secondaryBackgroundColor = '#B9F1C0'\n\
textColor = '#FFFFFF'\n\
font = 'sans serif'
" > ~/.streamlit/config.toml