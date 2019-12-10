import requests

params = {
	"facial_identification": [
		1.0, 1.0, 10.0
	],
	"avatar": "sadsa dsaddasd kjsahdkjsahkjd haskj",
	"big_avatar": "test big avatar"
	}

r = requests.post('http://system.whis.tech/api/employees/faceid/5dad95d39d156262dd7fc562', data = params)

print(r.status_code)



