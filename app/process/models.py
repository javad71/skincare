from django.db import models
from django.db.models import JSONField


class AcneQueue(models.Model):
    """
    RequestTable is a queue table
    """
    STATUS_CHOICES = (
        (0, 'Initial'),
        (1, 'Enqueue'),
        (2, 'Process'),
        (3, 'Terminate'),
        (4, 'Resend')
    )
    image = models.ImageField(max_length=255, blank=True, upload_to='inputs')
    time_enqueue = models.DateTimeField(null=True, blank=True)
    time_dequeue = models.DateTimeField(null=True, blank=True)
    status = models.SmallIntegerField(choices=STATUS_CHOICES, default=0)
    model = JSONField(null=True, blank=True)
    time_to_dead = models.IntegerField(null=True, blank=True, default=0)
    result = JSONField(null=True, blank=True)


class Log(models.Model):
    """
    Log all events in service
    """
    request_time = models.DateTimeField(null=True, blank=True)
    request_id = models.IntegerField(null=True, blank=True, default=0)
    request_event = models.CharField(max_length=50, blank=True, null=True)
    request_error_message = JSONField(null=True, blank=True)
