# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy import Request
from scrapy.pipelines.files import FilesPipeline


class PacktPipeline(FilesPipeline):
    def get_media_requests(self, item, info):
        req_list = []
        if 'pdf_url' in item and len(item['pdf_url']) > 0:
            req = Request(item['pdf_url'][0], priority=0)
            req.meta['book_name'] = item['title'][0]
            req.meta['ext'] = 'pdf'
            req_list.append(req)
        if 'epub_url' in item and len(item['epub_url']) > 0:
            req = Request(item['epub_url'][0], priority=0)
            req.meta['book_name'] = item['title'][0]
            req.meta['ext'] = 'epub'
            req_list.append(req)
        if 'mobi_url' in item and len(item['mobi_url']) > 0:
            req = Request(item['mobi_url'][0], priority=0)
            req.meta['book_name'] = item['title'][0]
            req.meta['ext'] = 'mobi'
            req_list.append(req)
        if 'code_url' in item and len(item['code_url']) > 0:
            req = Request(item['code_url'][0], priority=0)
            req.meta['book_name'] = item['title'][0]
            req.meta['ext'] = 'zip'
            req_list.append(req)
        return req_list

    def file_path(self, request, response=None, info=None):
        book_name = request.meta['book_name']
        ext = request.meta['ext']
        return book_name + '/' + book_name + '.' + ext
