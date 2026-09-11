import hashlib,json,sqlite3,subprocess,time
from pathlib import Path
p=Path('.private-vscode-session.vscdb')
c=sqlite3.connect(p)
rows=c.execute("select key,value from ItemTable where key like 'secret://%isDynamicAuthProvider%'").fetchall()
assert len(rows)==1
name,value=rows[0]
raw=bytes(json.loads(value)['data'])
assert raw[:3]==b'v10'
key=hashlib.pbkdf2_hmac('sha1',b'peanuts',b'saltysalt',1,16)
args=['openssl','enc','-aes-128-cbc','-K',key.hex(),'-iv',(b' '*16).hex()]
plain=subprocess.run(args+['-d'],input=raw[3:],capture_output=True,check=True).stdout
sessions=json.loads(plain)
assert len(sessions)==1 and sessions[0]['refresh_token']
original=sessions[0].copy()
sessions[0]['created_at']=int(time.time()*1000)-(sessions[0]['expires_in']+60)*1000
assert all(sessions[0][k]==v for k,v in original.items() if k!='created_at')
sealed=b'v10'+subprocess.run(args,input=json.dumps(sessions).encode(),capture_output=True,check=True).stdout
c.execute('update ItemTable set value=? where key=?',(json.dumps({'type':'Buffer','data':list(sealed)}),name));c.commit();c.close()
Path('after/vscode-refresh-fixture.json').write_text(json.dumps({'method':'Move only VS Code cached created_at into the past, with VS Code stopped; preserve real access/refresh tokens and all server settings','expires_in':original['expires_in'],'changed_field':'created_at','credentials_unchanged':True,'gateway_source_unchanged':True},indent=2))
print('Updated one cached timestamp; credentials and server unchanged')
