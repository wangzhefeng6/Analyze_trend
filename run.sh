today=`date -u "+%Y-%m-%d"`
cd daily_arxiv
scrapy crawl arxiv -o ../data/${today}.jsonl

cd ../ai
python improve.py --data ../data/${today}.jsonl

cd ../to_md
python convert.py --data ../data/${today}.jsonl

cd ..
python update_readme.py
