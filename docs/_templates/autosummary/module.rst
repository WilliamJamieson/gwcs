{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% if fullname == "gwcs.typing" %}
   .. rubric:: Type aliases

   {% for item in typing_type_aliases | sort %}
   .. autotype:: {{ item }}
   {% endfor %}
   {% endif %}

   {% block classes %}
   {% if all_classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :template: autosummary/class.rst
   {% for item in all_classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if all_exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree:
      :template: autosummary/class.rst
   {% for item in all_exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if all_functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
   {% for item in all_functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% if all_classes %}
.. inheritance-diagram:: {% for item in all_classes %}{{ fullname }}.{{ item }} {% endfor %}
   :parts: 1
{% endif %}
