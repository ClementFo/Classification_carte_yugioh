from django.http import HttpResponse


def prediction_machine_learning(request):

    html_http = '<h1>Visualisation</h1>'
    html_http += '<h2>Visualisation général des données</h2>'
    html_http += '<p>Ce premier graphique le nombre de carte par catégorie de carte</p>'
    return HttpResponse(html_http)
