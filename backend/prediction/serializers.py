from rest_framework import serializers


class GarbageInputSerializer(serializers.Serializer):
    garbage = serializers.ImageField()