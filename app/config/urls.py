from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from process.views import grabbing, get_queue_status, truncate_queue, get_log_by_id,\
    get_log_by_date, get_statistic_volumes, delete_contents_volumes

urlpatterns = [
    path("delete/", truncate_queue, name="truncate queue"),
    path("queue/", get_queue_status, name="get all data in queue"),
    path("filter-id/", get_log_by_id, name="get log by id"),
    path("filter-date/", get_log_by_date, name="get log by date"),
    path("statistic/", get_statistic_volumes, name="get statistic volumes"),
    path("delete-content/", delete_contents_volumes, name="delete contents of volumes"),
    path("grab/", grabbing, name="grab"),
    path("admin/", admin.site.urls),
]

if bool(settings.DEBUG):
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)