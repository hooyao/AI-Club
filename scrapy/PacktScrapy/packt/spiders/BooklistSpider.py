# -*- coding: utf-8 -*-
import re

import scrapy
from scrapy import Request, Selector
from scrapy.loader import ItemLoader

from packt.items import PacktBookItem


class BooklistSpider(scrapy.Spider):
    name = 'mybooks'
    allowed_domains = ['packtpub.com']
    start_urls = ['https://www.packtpub.com']

    PATTERN_DOWNLOAD_URL = re.compile(r'\/(ebook_download|code_download)\/\d+(\/(epub|pdf|mobi))?')

    def parse(self, response):
        request = scrapy.FormRequest.from_response(
            response,
            url='https://www.packtpub.com/',
            formdata={'email': 'your@email.com',
                      'password': 'password',
                      'op': 'Login',
                      'form_build_id': 'form-aeceb90f83fa58fb183d176c9ed45998',
                      'form_id': 'packt_user_login_form'
                      },
            callback=self.after_login

        )
        yield request

    def after_login(self, response):
        print(response.headers.getlist('Set-Cookie'))
        if 'Sign Out' in response.text:
            print('Login successful.')
            yield Request('https://www.packtpub.com/account/my-ebooks', callback=self.after_ebooks)
        else:
            print('Login Failed.')

    def after_ebooks(self, response):
        if 'My eBooks' in response.text:
            prod_lines = response.xpath("//div[contains(@class,'product-line')]")
            for prod_line in prod_lines:
                prod_line_html = prod_line.extract()
                sel = Selector(text=prod_line_html)
                title = sel.xpath("//div[contains(@class,'product-line')]/@title").extract_first()
                if title:
                    title = re.sub(r'\s\[eBook\]', '', title)
                    loader = ItemLoader(item=PacktBookItem(), response=response)
                    loader.add_value('title', title)
                    urls = sel.xpath("//div[contains(@class,'download-container')]/a/@href").extract()
                    urls = list(filter(lambda url: self.PATTERN_DOWNLOAD_URL.match(url), urls))
                    valid_urls = []
                    for url in urls:
                        if 'pdf' in url:
                            pdf_url = response.urljoin(url)
                            loader.add_value('pdf_url', pdf_url)
                            valid_urls.append(pdf_url)
                        elif 'epub' in url:
                            epub_url = response.urljoin(url)
                            loader.add_value('epub_url', epub_url)
                            valid_urls.append(epub_url)
                        elif 'mobi' in url:
                            mobi_url = response.urljoin(url)
                            loader.add_value('mobi_url', mobi_url)
                            valid_urls.append(mobi_url)
                        elif 'code' in url:
                            code_url = response.urljoin(url)
                            loader.add_value('code_url', code_url)
                            valid_urls.append(code_url)
                    loader.add_value('file_urls', valid_urls)
                    yield loader.load_item()
        else:
            print('Failed to get my-ebooks page.')
