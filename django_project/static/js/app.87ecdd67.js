(function(){"use strict";var e={6961:function(e,t,a){var r=a(9197),o=a(8473);const s={class:""};function l(e,t,a,r,l,n){const u=(0,o.up)("router-view");return(0,o.wg)(),(0,o.iD)("div",s,[(0,o.Wm)(u)])}var n={name:"App",components:{}},u=a(5312);const i=(0,u.Z)(n,[["render",l]]);var d=i,c=a(4731),m=a(4887),p=a.p+"static/img/logo.18e064fb.png";const f=e=>((0,o.dD)("data-v-b0a48e14"),e=e(),(0,o.Cn)(),e),g={class:"loginpage"},h={class:"login"},w=f((()=>(0,o._)("img",{src:p,alt:""},null,-1))),v={class:"topnav"},_={class:"form-group"},b={class:"form-group"},y={key:0,class:"tip2"},k=f((()=>(0,o._)("button",{class:"login-btn",type:"submit"},"登录",-1)));function C(e,t,a,s,l,n){const u=(0,o.up)("router-link");return(0,o.wg)(),(0,o.iD)("div",g,[(0,o._)("div",h,[(0,o.Wm)(u,{to:"/",class:"logo"},{default:(0,o.w5)((()=>[w])),_:1}),(0,o._)("div",v,[(0,o.Wm)(u,{to:"/about",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("文档")])),_:1}),(0,o.Wm)(u,{to:"/register",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("注册")])),_:1}),(0,o.Wm)(u,{to:"/",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("首页")])),_:1})]),(0,o._)("form",{onSubmit:t[2]||(t[2]=(0,r.iM)((function(){return e.login&&e.login(...arguments)}),["prevent"]))},[(0,o._)("div",_,[(0,o.wy)((0,o._)("input",{type:"text","onUpdate:modelValue":t[0]||(t[0]=t=>e.username=t),placeholder:"用户名"},null,512),[[r.nr,e.username]])]),(0,o._)("div",b,[(0,o.wy)((0,o._)("input",{type:"password","onUpdate:modelValue":t[1]||(t[1]=t=>e.password=t),placeholder:"密码"},null,512),[[r.nr,e.password]]),e.errorMessage?((0,o.wg)(),(0,o.iD)("div",y,(0,m.zw)(e.errorMessage),1)):(0,o.kq)("",!0)]),k,(0,o._)("p",null,[(0,o.Uk)("没有账号？去"),(0,o.Wm)(u,{to:"/register",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)(" 注册")])),_:1})])],32)])])}a(7658);var M=a(4188),W=a(9868),U=a.n(W),D=(0,o.aZ)({setup(){const e=(0,M.iH)(""),t=(0,M.iH)(""),a=(0,M.iH)(""),r=new FormData,o=async()=>{try{r.append("username",e.value),r.append("password",t.value);const a=await U().post("user/login/",r),o=a.headers.authorization.split(" ")[1];localStorage.removeItem("username"),localStorage.removeItem("token"),localStorage.setItem("username",e.value),localStorage.setItem("token",o),yt.push({path:"/workspace"})}catch(o){const e=o.response.data.error_msg;a.value=e}};return{username:e,password:t,errorMessage:a,login:o}}});const N=(0,u.Z)(D,[["render",C],["__scopeId","data-v-b0a48e14"]]);var S=N;const P=e=>((0,o.dD)("data-v-3c0d47d2"),e=e(),(0,o.Cn)(),e),j={class:"homepage"},T={class:"home"},x=P((()=>(0,o._)("img",{src:p,alt:""},null,-1))),I={class:"topnav"},F=P((()=>(0,o._)("h1",null,"Startup",-1))),V=P((()=>(0,o._)("div",{class:"introduction"},[(0,o.Uk)(" 分布式故障分类系统，"),(0,o._)("br"),(0,o.Uk)(" 接入多种神经网络和机器学习分布式故障分类算法，"),(0,o._)("br"),(0,o.Uk)(" 也可训练自己的专属模型，更快更精准！ ")],-1)));function q(e,t){const a=(0,o.up)("router-link");return(0,o.wg)(),(0,o.iD)("div",j,[(0,o._)("div",T,[(0,o.Wm)(a,{to:"/",class:"logo"},{default:(0,o.w5)((()=>[x])),_:1}),(0,o._)("div",I,[(0,o.Wm)(a,{to:"/about",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("文档")])),_:1}),(0,o.Wm)(a,{to:"/register",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("注册")])),_:1}),(0,o.Wm)(a,{to:"/login",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("登录")])),_:1})]),F,V,(0,o.Wm)(a,{to:"/workspace",class:"login"},{default:(0,o.w5)((()=>[(0,o.Uk)("创建项目")])),_:1})])])}const z={},O=(0,u.Z)(z,[["render",q],["__scopeId","data-v-3c0d47d2"]]);var H=O;const $=e=>((0,o.dD)("data-v-bbb8d43e"),e=e(),(0,o.Cn)(),e),R={class:"registerpage"},Z={class:"register"},L=$((()=>(0,o._)("img",{src:p,alt:""},null,-1))),A={class:"topnav"},B={class:"form-group"},E={class:"form-group"},Y={class:"form-group"},K={class:"form-group"},J=["src"],G=$((()=>(0,o._)("button",{class:"register-btn",type:"submit"},"注册",-1)));function Q(e,t,a,s,l,n){const u=(0,o.up)("router-link");return(0,o.wg)(),(0,o.iD)("div",R,[(0,o._)("div",Z,[(0,o.Wm)(u,{to:"/",class:"logo"},{default:(0,o.w5)((()=>[L])),_:1}),(0,o._)("div",A,[(0,o.Wm)(u,{to:"/about",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("文档")])),_:1}),(0,o.Wm)(u,{to:"/login",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("登录")])),_:1}),(0,o.Wm)(u,{to:"/",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("首页")])),_:1})]),(0,o._)("form",{onSubmit:t[5]||(t[5]=(0,r.iM)((function(){return e.onSubmit&&e.onSubmit(...arguments)}),["prevent"]))},[(0,o._)("div",B,[(0,o.wy)((0,o._)("input",{type:"text","onUpdate:modelValue":t[0]||(t[0]=t=>e.username=t),placeholder:"输入用户名"},null,512),[[r.nr,e.username]]),(0,o.wy)((0,o._)("div",{class:"tip1"},(0,m.zw)(e.errors.username),513),[[r.F8,!e.validUsername]])]),(0,o._)("div",E,[(0,o.wy)((0,o._)("input",{type:"password","onUpdate:modelValue":t[1]||(t[1]=t=>e.password=t),placeholder:"输入登录密码"},null,512),[[r.nr,e.password]]),(0,o.wy)((0,o._)("div",{class:"tip2"},(0,m.zw)(e.errors.password),513),[[r.F8,!e.validPassword]])]),(0,o._)("div",Y,[(0,o.wy)((0,o._)("input",{type:"password","onUpdate:modelValue":t[2]||(t[2]=t=>e.confirmPassword=t),placeholder:"再次输入登录密码"},null,512),[[r.nr,e.confirmPassword]]),(0,o.wy)((0,o._)("div",{class:"tip3"},(0,m.zw)(e.errors.confirmPassword),513),[[r.F8,""!==e.password&&""!==e.confirmPassword&&e.password!==e.confirmPassword]])]),(0,o._)("div",K,[(0,o.wy)((0,o._)("input",{type:"text","onUpdate:modelValue":t[3]||(t[3]=t=>e.securityCode=t),placeholder:"输入验证码",class:"security-code-input"},null,512),[[r.nr,e.securityCode]]),(0,o._)("img",{src:e.securityCodeUrl,alt:"验证码",onClick:t[4]||(t[4]=function(){return e.refreshSecurityCode&&e.refreshSecurityCode(...arguments)}),class:"captcha-img"},null,8,J),(0,o.wy)((0,o._)("div",{class:"tip4"},(0,m.zw)(e.errors.securityCode),513),[[r.F8,e.securityCodeInvalid]])]),G,(0,o._)("p",null,[(0,o.Uk)("已有账号？去"),(0,o.Wm)(u,{to:"/login",class:"routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)(" 登录")])),_:1})])],32)])])}var X=a(7016),ee=(0,o.aZ)({setup(){const e=(0,M.iH)(""),t=(0,M.iH)(""),a=(0,M.iH)(""),r=(0,M.iH)(""),s=(0,M.iH)(!1),l=new FormData,n={username:"",password:"",confirmPassword:"",securityCode:""},u=(0,M.iH)(""),i=(0,M.iH)(!1),d=(0,M.iH)(!1),c=(0,M.iH)(!1);(0,o.YP)(e,(async e=>{if(""===e.trim())return;const t=e.trim(),a=/^\w{6,20}$/.test(t);if(!a)return n.username="用户名为长度为6-20位的数字和字母的组合",void(i.value=!1);i.value=!0,n.username=""})),(0,o.YP)(t,(e=>{if(""===e.trim())return;const t=e;d.value=!!t&&/^[^\s]*(\s+[^\s]*){0}$/.test(t)&&/(?=.*[0-9])(?=.*[a-zA-Z])(?=.*[^a-zA-Z0-9])[^\s]{8,16}/.test(t),d.value?(d.value=!0,n.password=""):n.password="密码为8-16位数字、字母和特殊字符（不包括空格）"})),(0,o.YP)(a,(e=>{if(""===e.trim())return;const a=e.trim();c.value=!!a&&a===t.value})),(0,o.YP)([t,a],(e=>{let[t,a]=e;t&&a?c.value=!!t&&t===a:(c.value=!1,n.confirmPassword="两次输入的密码不一致！")}));const m=()=>{U().get("/user/refresh_code").then((e=>{u.value=e.data.image_url,s.value=!1,r.value=""})).catch((e=>{const t=e.response.data.error_msg;(0,X.bM)({type:"error",message:t})}))},p=()=>{i.value&&d.value&&c.value&&r.value&&(l.append("username",e.value),l.append("password",t.value),l.append("code",r.value),U().post("user/register/",l).then((e=>{console.log(e),yt.push("/login")})).catch((e=>{if(e.response){const t=e.response.data.error_msg;(0,X.bM)({type:"warning",message:t})}})))};return m(),{username:e,password:t,confirmPassword:a,securityCode:r,errors:n,securityCodeUrl:u,validUsername:i,validPassword:d,validConfirmPassword:c,securityCodeInvalid:s,refreshSecurityCode:m,onSubmit:p}}});const te=(0,u.Z)(ee,[["render",Q],["__scopeId","data-v-bbb8d43e"]]);var ae=te;function re(e,t){return(0,o.wg)(),(0,o.iD)("div",null,"使用文档")}const oe={},se=(0,u.Z)(oe,[["render",re]]);var le=se;const ne=e=>((0,o.dD)("data-v-3e479e8a"),e=e(),(0,o.Cn)(),e),ue={class:"WorkPage"},ie=ne((()=>(0,o._)("img",{src:p,alt:""},null,-1))),de={class:"username"},ce=ne((()=>(0,o._)("div",{class:"intomain"},null,-1)));function me(e,t,a,r,s,l){const n=(0,o.up)("router-link"),u=(0,o.up)("el-header"),i=(0,o.up)("el-aside"),d=(0,o.up)("router-view"),c=(0,o.up)("el-main"),p=(0,o.up)("el-container");return(0,o.wg)(),(0,o.iD)("div",ue,[(0,o.Wm)(p,null,{default:(0,o.w5)((()=>[(0,o.Wm)(u,null,{default:(0,o.w5)((()=>[(0,o.Wm)(n,{to:"/",class:"logo routerlink"},{default:(0,o.w5)((()=>[ie])),_:1}),(0,o.Wm)(n,{to:"/about",class:"about routerlink"},{default:(0,o.w5)((()=>[(0,o.Uk)("使用文档")])),_:1}),(0,o._)("span",de,[(0,o.Uk)("欢迎，"),(0,o._)("span",{onClick:t[0]||(t[0]=function(){return e.handleLogout&&e.handleLogout(...arguments)}),class:"logout"},(0,m.zw)(e.username),1)])])),_:1}),(0,o.Wm)(p,null,{default:(0,o.w5)((()=>[(0,o.Wm)(i,{width:"15%"},{default:(0,o.w5)((()=>[(0,o._)("div",{onClick:t[1]||(t[1]=function(){return e.totrain&&e.totrain(...arguments)}),"exact-active-class":"active","active-class":"active"},"训练中心"),(0,o._)("div",{onClick:t[2]||(t[2]=function(){return e.totest&&e.totest(...arguments)}),"exact-active-class":"active","active-class":"active"},"模型预测")])),_:1}),(0,o.Wm)(c,null,{default:(0,o.w5)((()=>[ce,(0,o.Wm)(d)])),_:1})])),_:1})])),_:1})])}a(2801);const pe=U().create({});pe.interceptors.request.use((e=>{const t=localStorage.getItem("token");return t&&(e.headers.Authorization=`Token ${t}`),e}),(e=>Promise.reject(e)));U().create({});const{CancelToken:fe}=U();async function ge(){try{return await pe.post("user/logout/"),localStorage.removeItem("token"),delete pe.defaults.headers.common.Authorization,localStorage.removeItem("username"),!0}catch(e){return console.error(e),!1}}var he=(0,o.aZ)({setup(){const e=localStorage.getItem("username"),t=async()=>{const e=await ge();e?yt.push("./login"):alert("退出失败！请稍后重试！")},a=()=>{yt.push("/TrainModel")},r=()=>{yt.push("/ModelPrediction")};return{username:e,handleLogout:t,totrain:a,totest:r}}});const we=(0,u.Z)(he,[["render",me],["__scopeId","data-v-3e479e8a"]]);var ve=we;const _e=e=>((0,o.dD)("data-v-32288e0a"),e=e(),(0,o.Cn)(),e),be=_e((()=>(0,o._)("h1",null,"404 Not Found!",-1)));function ye(e,t,a,r,s,l){const n=(0,o.up)("router-link");return(0,o.wg)(),(0,o.iD)("div",null,[be,(0,o._)("p",null,[(0,o.Uk)((0,m.zw)(e.count)+"秒后将自动跳转至",1),(0,o.Wm)(n,{to:"/"},{default:(0,o.w5)((()=>[(0,o.Uk)("首页")])),_:1})])])}var ke=(0,o.aZ)({setup(){let e=(0,M.iH)(5);const t=setInterval((()=>{e.value--}),1e3);return(0,o.YP)(e,(e=>{0===e&&(clearInterval(t),yt.push("/"))})),{count:e}}});const Ce=(0,u.Z)(ke,[["render",ye],["__scopeId","data-v-32288e0a"]]);var Me=Ce;const We=e=>((0,o.dD)("data-v-cff8101e"),e=e(),(0,o.Cn)(),e),Ue={class:"container"},De=We((()=>(0,o._)("h3",null,"创建新的训练",-1))),Ne={class:"data"},Se=We((()=>(0,o._)("div",{class:"el-upload__tip"},"支持 .csv 格式文件",-1))),Pe=We((()=>(0,o._)("h3",{style:{"margin-top":"20px","margin-bottom":"10px"}},"训练进程展示",-1))),je={key:0},Te={key:1},xe={key:2},Ie={key:3},Fe={key:4},Ve={key:0},qe={key:0};function ze(e,t,a,r,s,l){const n=(0,o.up)("el-button"),u=(0,o.up)("el-upload"),i=(0,o.up)("el-form-item"),d=(0,o.up)("el-form"),c=(0,o.up)("el-radio"),p=(0,o.up)("el-input"),f=(0,o.up)("el-table-column"),g=(0,o.up)("el-table");return(0,o.wg)(),(0,o.iD)("div",Ue,[De,(0,o._)("div",Ne,[(0,o.Wm)(d,null,{default:(0,o.w5)((()=>[(0,o.Wm)(i,{label:"上传文件"},{default:(0,o.w5)((()=>[(0,o.Wm)(u,{ref:"fileUpload","on-change":r.handleFileChange,"show-file-list":!1,"auto-upload":!1,accept:".csv",multiple:!1},{default:(0,o.w5)((()=>[(0,o.Wm)(n,{size:"small",type:"primary",class:"choseFile"},{default:(0,o.w5)((()=>[(0,o.Uk)("选择文件")])),_:1}),Se])),_:1},8,["on-change"])])),_:1})])),_:1}),(0,o.Wm)(d,{model:r.form,ref:"createForm",style:{"max-width":"600px"}},{default:(0,o.w5)((()=>[(0,o.Wm)(i,{label:"选择算法"},{default:(0,o.w5)((()=>[((0,o.wg)(!0),(0,o.iD)(o.HY,null,(0,o.Ko)(r.models,(e=>((0,o.wg)(),(0,o.j4)(c,{modelValue:r.form.modelName,"onUpdate:modelValue":t[0]||(t[0]=e=>r.form.modelName=e),label:e.name,key:e.name},{default:(0,o.w5)((()=>[(0,o.Uk)((0,m.zw)(e.name),1)])),_:2},1032,["modelValue","label"])))),128))])),_:1}),(0,o.Wm)(i,{label:"模型名称"},{default:(0,o.w5)((()=>[(0,o.Wm)(p,{modelValue:r.form.userModelName,"onUpdate:modelValue":t[1]||(t[1]=e=>r.form.userModelName=e),placeholder:"请输入模型名称",onBlur:r.checkModelName},null,8,["modelValue","onBlur"])])),_:1}),(0,o.Wm)(i,{label:"特征值数"},{default:(0,o.w5)((()=>[(0,o.Wm)(p,{type:"number",modelValue:r.form.featuresNum,"onUpdate:modelValue":t[2]||(t[2]=e=>r.form.featuresNum=e),placeholder:"请输入特征值数"},null,8,["modelValue"])])),_:1}),(0,o.Wm)(i,{label:"标签范围"},{default:(0,o.w5)((()=>[(0,o.Wm)(p,{type:"number",modelValue:r.form.labelsNum,"onUpdate:modelValue":t[3]||(t[3]=e=>r.form.labelsNum=e),placeholder:"请输入标签范围"},null,8,["modelValue"])])),_:1}),(0,o.Wm)(i,null,{default:(0,o.w5)((()=>[(0,o.Wm)(n,{type:"primary",onClick:r.createTrain},{default:(0,o.w5)((()=>[(0,o.Uk)("确定创建")])),_:1},8,["onClick"]),(0,o.Wm)(n,{onClick:r.cancel},{default:(0,o.w5)((()=>[(0,o.Uk)("取消创建")])),_:1},8,["onClick"])])),_:1})])),_:1},8,["model"])]),Pe,(0,o.Wm)(g,{data:r.trainTasks,style:{"max-width":"1024px"},"header-cell-style":{backgroundColor:"#f5f7fa"}},{default:(0,o.w5)((()=>[(0,o.Wm)(f,{prop:"name",label:"训练任务"}),(0,o.Wm)(f,{prop:"status",label:"训练状态"},{default:(0,o.w5)((e=>{let{row:t}=e;return[""===t.status?((0,o.wg)(),(0,o.iD)("span",je,"等待中")):"in_progress"===t.status?((0,o.wg)(),(0,o.iD)("span",Te,"训练中")):"terminated"===t.status?((0,o.wg)(),(0,o.iD)("span",xe,"已取消")):"exception"===t.status?((0,o.wg)(),(0,o.iD)("span",Ie,"训练失败")):"success"===t.status?((0,o.wg)(),(0,o.iD)("span",Fe,"已完成")):(0,o.kq)("",!0)]})),_:1}),(0,o.Wm)(f,{prop:"startTime",label:"开始时间"}),(0,o.Wm)(f,{label:"Accuracy"},{default:(0,o.w5)((e=>{let{row:t}=e;return["success"===t.status?((0,o.wg)(),(0,o.iD)("span",Ve,(0,m.zw)(t.accuracy),1)):(0,o.kq)("",!0)]})),_:1}),(0,o.Wm)(f,{label:"MacroF1"},{default:(0,o.w5)((e=>{let{row:t}=e;return["success"===t.status?((0,o.wg)(),(0,o.iD)("span",qe,(0,m.zw)(t.MacroF1),1)):(0,o.kq)("",!0)]})),_:1}),(0,o.Wm)(f,{label:"操作"},{default:(0,o.w5)((e=>{let{row:t}=e;return["success"===t.status?((0,o.wg)(),(0,o.j4)(n,{key:0,type:"primary",href:r.modelURL,onClick:e=>r.downloadModel(t.id)},{default:(0,o.w5)((()=>[(0,o.Uk)("下载模型")])),_:2},1032,["href","onClick"])):(0,o.kq)("",!0)]})),_:1})])),_:1},8,["data"])])}a(2062);var Oe={name:"TrainingPage",setup(){const e=(0,M.qj)([{name:"SVM"},{name:"RandomForest_K"},{name:"CNN"},{name:"SVM_K"}]);let t=new FormData,a=(0,M.qj)([]),r=(0,M.qj)({modelName:"",userModelName:"",featuresNum:"",labelsNum:""}),o=null,s=(0,M.iH)(null);const l=e=>{t.delete("file"),o=e.name,t.append("file",e.raw,e.name),console.log(e.raw)},n=()=>{if(""===r.userModelName)(0,X.bM)({type:"error",message:"模型名不能为空"});else if(!/^\w{6,20}$/.test(r.userModelName))return void(0,X.bM)({type:"error",message:"模型名称由6-20位字母数字组成"})},u=()=>{const e=new Date,t=String(e.getMonth()+1).padStart(2,"0"),a=String(e.getDate()).padStart(2,"0"),r=String(e.getHours()).padStart(2,"0"),o=String(e.getMinutes()).padStart(2,"0"),s=`${t}-${a} ${r}:${o}`;return s.toString()},i=()=>{if(""===r.modelName||""===r.userModelName||""===r.featuresNum||""===r.labelsNum||!o)return void(0,X.bM)({type:"error",message:"请填写完整的训练任务信息"});a.unshift((0,M.qj)({id:null,name:r.userModelName,status:"in_progress",startTime:u(),accuracy:0,MacroF1:0})),console.log(a);const e=u();t.append("create_time",e),t.append("algorithm",r.modelName),t.append("model_name",r.userModelName),t.append("featuresNum",r.featuresNum),t.append("labelsNum",r.labelsNum),console.log(t.get("file")),pe.post("upload/train/",t,{headers:{"Content-Type":"multipart/form-data"}}).then((e=>{d(),console.log("--------------------------","yes,train了"),(0,X.bM)({type:"success",message:"创建训练任务成功"});const t=e.data.data,a=(0,M.qj)({id:t.id,name:t.model_name,status:t.status,startTime:t.create_time,accuracy:0,MacroF1:0});this.trainTasks.push(a),r.userModelName="",r.featuresNum="",r.labelsNum="",r.modelName="",o="";const{fileUpload:s}=this.$refs;s&&s.clearFiles()})).catch((e=>{const t=e.response.status;if(401===t)(0,X.bM)({type:"error",message:"未校验，请重新登录！"}),yt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,X.bM)({type:"error",message:t}),console.error(e)}}})).finally((()=>(c(),t.delete("name"),t.delete("file"),t.delete("algorithm"),t.delete("featuresNum"),t.delete("labelsNum"),o="",new Promise((()=>{})))))},d=()=>{pe.get("upload/find_train/").then((e=>{const t=e.data.data;a.splice(0),t.forEach((e=>{a.push((0,M.qj)({id:e.train_id,name:e.model_name,status:e.status,startTime:e.create_time,accuracy:e.Accuracy,MacroF1:e.MacroF1}))})),a.reverse()})).catch((e=>{const t=e.response.status;if(401===t)(0,X.bM)({type:"error",message:"未校验，请重新登录！"}),yt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,X.bM)({type:"error",message:t}),console.error(e)}}}))},c=()=>{if(r.userModelName="",r.featuresNum="",r.labelsNum="",r.modelName="",o="",t.delete("file"),s.value){const{fileUpload:e}=s.value;e&&e.clearFiles()}},m=(0,M.iH)(""),p=e=>{const t=new FormData;t.append("taskid",e),pe.post("/upload/model_download/",t,{responseType:"blob"}).then((e=>{const t=e.headers["content-disposition"].match(/filename="([^"]+)"/)[1].replace(/"/g,""),a=URL.createObjectURL(e.data),r=document.createElement("a");r.style.display="none",r.href=a,r.setAttribute("download",t),document.body.appendChild(r),r.click(),URL.revokeObjectURL(a),document.body.removeChild(r)})).catch((e=>{const t=e.response.status;if(401===t)(0,X.bM)({type:"error",message:"未校验，请重新登录！"}),yt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,X.bM)({type:"error",message:t}),console.error(e)}}}))};return d(),{models:e,formData:t,downloadModel:p,modelURL:m,trainTasks:a,form:r,fileUploadRef:s,handleFileChange:l,createTrain:i,getTrainTasks:d,cancel:c,checkModelName:n}}};const He=(0,u.Z)(Oe,[["render",ze],["__scopeId","data-v-cff8101e"]]);var $e=He;const Re=e=>((0,o.dD)("data-v-3abe2739"),e=e(),(0,o.Cn)(),e),Ze={class:"container"},Le=Re((()=>(0,o._)("h3",null,"开始预测",-1))),Ae={class:"data"},Be=Re((()=>(0,o._)("div",{class:"el-upload__tip"},"支持 .csv 格式文件",-1))),Ee={class:"modelCard"},Ye={class:"choseModel"},Ke=Re((()=>(0,o._)("span",{style:{"font-size":"14px",color:"var(--el-text-color-regular)","margin-right":"12px"}},"选择模型",-1))),Je={key:0,class:"showModel"},Ge={class:"card-item"},Qe={class:"card-content-item"},Xe=Re((()=>(0,o._)("h3",{style:{"margin-top":"20px","margin-bottom":"10px"}},"测试结果",-1))),et={key:0},tt={key:1},at={key:2},rt={key:3},ot={key:4};function st(e,t,a,r,s,l){const n=(0,o.up)("el-button"),u=(0,o.up)("el-upload"),i=(0,o.up)("el-form-item"),d=(0,o.up)("el-form"),c=(0,o.up)("el-option"),p=(0,o.up)("el-select"),f=(0,o.up)("el-card"),g=(0,o.up)("el-input"),h=(0,o.up)("el-table-column"),w=(0,o.up)("el-table");return(0,o.wg)(),(0,o.iD)("div",Ze,[Le,(0,o._)("div",Ae,[(0,o.Wm)(d,null,{default:(0,o.w5)((()=>[(0,o.Wm)(i,{label:"上传文件"},{default:(0,o.w5)((()=>[(0,o.Wm)(u,{ref:"fileUpload","on-change":r.handleFileChange,"show-file-list":!1,"auto-upload":!1,accept:".csv",multiple:!1},{default:(0,o.w5)((()=>[(0,o.Wm)(n,{size:"small",type:"primary",class:"choseFile"},{default:(0,o.w5)((()=>[(0,o.Uk)("选择文件")])),_:1}),Be])),_:1},8,["on-change"])])),_:1})])),_:1}),(0,o._)("div",Ee,[(0,o._)("div",Ye,[Ke,(0,o.Wm)(p,{label:"选择模型",modelValue:r.form.modelid,"onUpdate:modelValue":t[0]||(t[0]=e=>r.form.modelid=e),onChange:t[1]||(t[1]=e=>r.handleModelChange(r.form.modelid))},{default:(0,o.w5)((()=>[((0,o.wg)(!0),(0,o.iD)(o.HY,null,(0,o.Ko)(r.models,(e=>((0,o.wg)(),(0,o.j4)(c,{value:e.id,label:e.name,key:e.id},{default:(0,o.w5)((()=>[(0,o.Uk)((0,m.zw)(e.name),1)])),_:2},1032,["value","label"])))),128))])),_:1},8,["modelValue"])]),0!==Object.keys(r.selectedModel).length?((0,o.wg)(),(0,o.iD)("div",Je,[(0,o.Wm)(f,{shadow:"hover"},{default:(0,o.w5)((()=>[(0,o._)("div",Ge,[(0,o._)("div",null,"模型名称："+(0,m.zw)(r.selectedModel.name),1),(0,o._)("div",Qe,"特征值数: "+(0,m.zw)(r.selectedModel.featuresNum),1)])])),_:1})])):(0,o.kq)("",!0)]),(0,o.Wm)(d,null,{default:(0,o.w5)((()=>[(0,o.Wm)(i,{label:"特征值数"},{default:(0,o.w5)((()=>[(0,o.Wm)(g,{type:"number",modelValue:r.form.featuresNum,"onUpdate:modelValue":t[2]||(t[2]=e=>r.form.featuresNum=e),placeholder:"请输入特征值数",style:{width:"215px"}},null,8,["modelValue"])])),_:1}),(0,o.Wm)(i,null,{default:(0,o.w5)((()=>[(0,o.Wm)(n,{type:"primary",onClick:r.createPrediction},{default:(0,o.w5)((()=>[(0,o.Uk)("确定创建")])),_:1},8,["onClick"]),(0,o.Wm)(n,{onClick:r.cancel},{default:(0,o.w5)((()=>[(0,o.Uk)("取消创建")])),_:1},8,["onClick"])])),_:1})])),_:1})]),Xe,(0,o.Wm)(w,{data:r.predictionTasks,style:{"max-width":"1024px"},"header-cell-style":{backgroundColor:"#f5f7fa"}},{default:(0,o.w5)((()=>[(0,o.Wm)(h,{prop:"name",label:"预测任务"}),(0,o.Wm)(h,{prop:"status",label:"预测状态"},{default:(0,o.w5)((e=>{let{row:t}=e;return[""===t.status?((0,o.wg)(),(0,o.iD)("span",et,"等待中")):"in_progress"===t.status?((0,o.wg)(),(0,o.iD)("span",tt,"预测中")):"terminated"===t.status?((0,o.wg)(),(0,o.iD)("span",at,"已取消")):"exception"===t.status?((0,o.wg)(),(0,o.iD)("span",rt,"预测失败")):"success"===t.status?((0,o.wg)(),(0,o.iD)("span",ot,"已完成")):(0,o.kq)("",!0)]})),_:1}),(0,o.Wm)(h,{prop:"startTime",label:"开始时间"}),(0,o.Wm)(h,{label:"操作"},{default:(0,o.w5)((e=>{let{row:t}=e;return["success"===t.status?((0,o.wg)(),(0,o.j4)(n,{key:0,type:"primary",onClick:e=>r.showCharts(t.id)},{default:(0,o.w5)((()=>[(0,o.Uk)("可视化展示")])),_:2},1032,["onClick"])):(0,o.kq)("",!0)]})),_:1})])),_:1},8,["data"])])}var lt={setup(){let e=(0,M.qj)([]),t=new FormData,a=(0,M.qj)([]),r=(0,M.qj)({modelid:"",featuresNum:""}),o=(0,M.iH)({});const s=()=>{pe.get("upload/find_model/").then((t=>{const a=t.data.data;console.log(a),a.forEach((t=>{const a=(0,M.qj)({id:t.model_id,name:t.model_name,featuresNum:t.featuresNum});e.push(a)})),console.log("------------------------",e)})).catch((e=>{const t=e.response.status;if(401===t)(0,X.bM)({type:"error",message:"未校验，请重新登录！"}),yt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,X.bM)({type:"error",message:t}),console.error(e)}}}))};function l(t){o.value=e.find((e=>e.id===t))}const n=e=>{yt.push({path:`/ResultsPage/${e}`})};let u=null,i=(0,M.iH)(null);const d=e=>{t.delete("file"),u=e.name,t.append("file",e.raw,e.name),console.log(e.raw)},c=()=>{const e=new Date,t=String(e.getMonth()+1).padStart(2,"0"),a=String(e.getDate()).padStart(2,"0"),r=String(e.getHours()).padStart(2,"0"),o=String(e.getMinutes()).padStart(2,"0"),s=`${t}-${a} ${r}:${o}`;return s.toString()},m=()=>{if(""===r.modelid||""===r.featuresNum||!u)return void(0,X.bM)({type:"error",message:"请填写完整的训练任务信息"});t.append("model_id",r.modelid),t.append("featuresNum",r.featuresNum),console.log(t.get("file")),p(),a.unshift((0,M.qj)({id:null,name:u,status:"in_progress",startTime:c()}));const e=c();t.append("create_time",e),pe.post("upload/test/",t,{headers:{"Content-Type":"multipart/form-data"}}).then((e=>{p(),(0,X.bM)({type:"success",message:"创建预测任务成功"});const t=e.data.data;a.push((0,M.qj)({id:t.id,name:u,status:"in_progress",startTime:t.create_time})),r.featuresNum="",r.modelid="",u="",o.value={}})).catch((e=>{const t=e.response.status;if(401===t)(0,X.bM)({type:"error",message:"未校验，请重新登录！"}),yt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,X.bM)({type:"error",message:t}),console.error(e)}}})).finally((()=>(f(),t.delete("model_id"),t.delete("file"),t.delete("featuresNum"),u="",new Promise((()=>{})))))},p=()=>{pe.get("upload/find_test/").then((e=>{a.splice(0);const t=e.data.data;t.forEach((e=>{a.push((0,M.qj)({id:e.test_id,name:e.name,status:e.status,startTime:e.create_time}))})),a.reverse()})).catch((e=>{const t=e.response.status;if(401===t)(0,X.bM)({type:"error",message:"未校验，请重新登录！"}),yt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,X.bM)({type:"error",message:t}),console.error(e)}}}))},f=()=>{r.modelid="",r.featuresNum="",o.value={},u="",t.delete("file")};return s(),p(),{models:e,formData:t,predictionTasks:a,fileUploadRef:i,form:r,selectedModel:o,showCharts:n,handleFileChange:d,handleModelChange:l,createPrediction:m,getPredictionTasks:p,cancel:f}}};const nt=(0,u.Z)(lt,[["render",st],["__scopeId","data-v-3abe2739"]]);var ut=nt;const it={class:"container"},dt={class:"download-btns"},ct={class:"show"},mt={class:"table-container"},pt={ref:"pieCanvas",class:"pieCanvas"};function ft(e,t,a,r,s,l){const n=(0,o.up)("el-button"),u=(0,o.up)("el-table-column"),i=(0,o.up)("el-table");return(0,o.wg)(),(0,o.iD)("div",it,[(0,o._)("div",dt,[(0,o.Wm)(n,{type:"primary",onClick:l.downloadTableData},{default:(0,o.w5)((()=>[(0,o.Uk)(" 下载结果 ")])),_:1},8,["onClick"]),(0,o.Wm)(n,{type:"primary",onClick:l.downloadPieChart,style:{"margin-left":"65px"}},{default:(0,o.w5)((()=>[(0,o.Uk)(" 下载图形 ")])),_:1},8,["onClick"])]),(0,o._)("div",ct,[(0,o._)("div",mt,[(0,o.Wm)(i,{data:s.tableData,class:"table"},{default:(0,o.w5)((()=>[(0,o.Wm)(u,{prop:"sampleId",label:"Sample Id"}),(0,o.Wm)(u,{prop:"label",label:"Label"})])),_:1},8,["data"])]),(0,o._)("canvas",pt,null,512)])])}a(1439),a(7585),a(5315);var gt=a(2305),ht={name:"MyComponent",props:["taskId"],data(){return{tableData:[],pieChartData:[],pieChart:null}},mounted(){const e=new FormData;e.append("taskid",parseInt(this.taskId)),this.$nextTick((()=>{this.$refs.pieCanvas&&this.renderPieChart()})),pe.post("/upload/result_download/",e).then((e=>{const t=e.data.data;let a=t[0],r=t[1];const o=[],s=[];for(let l in a)o.push({sampleId:l,label:a[l]});for(let l in r)s.push({label:l,value:r[l]});this.tableData=(0,M.qj)(o),this.pieChartData=(0,M.qj)(s),this.renderPieChart()})).catch((e=>{const t=e.response.status;if(401===t)(0,X.bM)({type:"error",message:"未校验，请重新登录！"}),yt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,X.bM)({type:"error",message:t}),console.error(e)}}}))},methods:{renderPieChart(){if(!this.$refs.pieCanvas)return;this.pieChart=gt.S1(this.$refs.pieCanvas,null,{devicePixelRatio:2});const e=this.pieChartData.map((e=>({value:parseInt(e.value),name:e.label}))),t={grid:{width:"60%",height:"60%",left:"center"},series:[{type:"pie",data:e,label:{show:!0,formatter:"{b}: {c}",textStyle:{fontSize:14}},emphasis:{label:{show:!0,textStyle:{fontSize:1}}}}]};this.pieChart.setOption(t)},downloadTableData(){const e=this.arrayToCSV(this.tableData),t=new Blob([e],{type:"text/csv;charset=utf-8;"}),a=document.createElement("a");a.href=URL.createObjectURL(t),a.download="table-data.csv",a.click()},downloadPieChart(){if(!this.$refs.pieCanvas)return;const e=this.$refs.pieCanvas.toDataURL("image/png"),t=this.dataURItoBlob(e),a=document.createElement("a");a.href=URL.createObjectURL(t),a.download="pie-chart.png",a.click()},dataURItoBlob(e){const t=atob(e.split(",")[1]),a=[];for(let o=0;o<t.length;o++)a.push(t.charCodeAt(o));const r=e.split(",")[0].split(":")[1].split(";")[0];return new Blob([new Uint8Array(a)],{type:r})},arrayToCSV(e){const t=[],a=Object.keys(e[0]);t.push(a.join(","));for(const r of e){const e=a.map((e=>{const t=null===r[e]||void 0===r[e]?"":r[e];return JSON.stringify(t)}));t.push(e.join(","))}return t.join("\n")}}};const wt=(0,u.Z)(ht,[["render",ft],["__scopeId","data-v-63633d4d"]]);var vt=wt;const _t=[{path:"/:pathMatch(.*)*",component:Me},{path:"/login",component:S},{path:"/",component:H},{path:"/register",component:ae},{path:"/about",component:le},{path:"/workspace",component:ve,children:[{path:"/TrainModel",component:$e},{path:"/ResultsPage/:taskId",component:vt,props:!0},{path:"/ModelPrediction",component:ut}]}],bt=(0,c.p7)({history:(0,c.PO)("/"),routes:_t});var yt=bt,kt=a(502);a(2834);const Ct=localStorage.getItem("token");pe.defaults.headers.common.Authorization=Ct?`Bearer ${Ct}`:"",(0,r.ri)(d).use(yt).use(kt.Z).mount("#app")}},t={};function a(r){var o=t[r];if(void 0!==o)return o.exports;var s=t[r]={exports:{}};return e[r].call(s.exports,s,s.exports,a),s.exports}a.m=e,function(){var e=[];a.O=function(t,r,o,s){if(!r){var l=1/0;for(d=0;d<e.length;d++){r=e[d][0],o=e[d][1],s=e[d][2];for(var n=!0,u=0;u<r.length;u++)(!1&s||l>=s)&&Object.keys(a.O).every((function(e){return a.O[e](r[u])}))?r.splice(u--,1):(n=!1,s<l&&(l=s));if(n){e.splice(d--,1);var i=o();void 0!==i&&(t=i)}}return t}s=s||0;for(var d=e.length;d>0&&e[d-1][2]>s;d--)e[d]=e[d-1];e[d]=[r,o,s]}}(),function(){a.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return a.d(t,{a:t}),t}}(),function(){a.d=function(e,t){for(var r in t)a.o(t,r)&&!a.o(e,r)&&Object.defineProperty(e,r,{enumerable:!0,get:t[r]})}}(),function(){a.g=function(){if("object"===typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"===typeof window)return window}}()}(),function(){a.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)}}(),function(){a.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})}}(),function(){a.p="/"}(),function(){var e={143:0};a.O.j=function(t){return 0===e[t]};var t=function(t,r){var o,s,l=r[0],n=r[1],u=r[2],i=0;if(l.some((function(t){return 0!==e[t]}))){for(o in n)a.o(n,o)&&(a.m[o]=n[o]);if(u)var d=u(a)}for(t&&t(r);i<l.length;i++)s=l[i],a.o(e,s)&&e[s]&&e[s][0](),e[s]=0;return a.O(d)},r=self["webpackChunkfaultc"]=self["webpackChunkfaultc"]||[];r.forEach(t.bind(null,0)),r.push=t.bind(null,r.push.bind(r))}();var r=a.O(void 0,[998],(function(){return a(6961)}));r=a.O(r)})();
//# sourceMappingURL=app.87ecdd67.js.map