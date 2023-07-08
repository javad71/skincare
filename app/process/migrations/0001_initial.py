# Generated by Django 3.2.15 on 2022-08-22 07:01

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AcneQueue',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(blank=True, max_length=255, upload_to='inputs')),
                ('time_enqueue', models.DateTimeField(blank=True, null=True)),
                ('time_dequeue', models.DateTimeField(blank=True, null=True)),
                ('status', models.SmallIntegerField(choices=[(0, 'Initial'), (1, 'Enqueue'), (2, 'Process'), (3, 'Terminate'), (4, 'Resend')], default=0)),
                ('model', models.JSONField(blank=True, null=True)),
                ('time_to_dead', models.IntegerField(blank=True, default=0, null=True)),
                ('result', models.JSONField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Log',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('request_time', models.DateTimeField(blank=True, null=True)),
                ('request_id', models.IntegerField(blank=True, default=0, null=True)),
                ('request_event', models.CharField(blank=True, max_length=50, null=True)),
                ('request_error_message', models.JSONField(blank=True, null=True)),
            ],
        ),
    ]
