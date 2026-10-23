use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Expr, ExprArray, Fields, Ident, Lit, LitStr, Path, Type, Visibility,
    meta::ParseNestedMeta, parse_macro_input,
};

/// `litellm_setting = "x"`: `litellm.x` in Python and `litellm_settings.x` in config.yaml, which the proxy copies with `setattr`.
#[proc_macro_derive(Layered, attributes(layer))]
pub fn derive_layered(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[derive(Default)]
struct FieldAttributes {
    env: Vec<LitStr>,
    read: Option<Expr>,
    from_env: Option<Path>,
    litellm_param: Vec<LitStr>,
    deserialize_with: Option<Path>,
    litellm_setting: Option<LitStr>,
    secret: bool,
    keep_blank: bool,
    setting: Option<LitStr>,
}

struct LayeredField {
    vis: Visibility,
    ident: Ident,
    ty: Type,
    attributes: FieldAttributes,
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(input, "Layered needs a struct"));
    };
    let Fields::Named(named) = &data.fields else {
        return Err(syn::Error::new_spanned(input, "Layered needs named fields"));
    };
    let fields = named
        .named
        .iter()
        .map(|field| {
            Ok(LayeredField {
                vis: field.vis.clone(),
                ident: field.ident.clone().expect("named field"),
                ty: field.ty.clone(),
                attributes: field_attributes(field)?,
            })
        })
        .collect::<syn::Result<Vec<_>>>()?;

    let name = &input.ident;
    let vis = &input.vis;
    let partial = format_ident!("{name}Partial");
    let layered = quote!(::litellm_layered);
    let private = quote!(#layered::__private);

    let partial_fields = fields.iter().map(|field| {
        let LayeredField { vis, ident, ty, .. } = field;
        quote!(#vis #ident: ::core::option::Option<#ty>)
    });
    let debug_fields = fields.iter().map(|field| {
        let ident = &field.ident;
        let label = ident.to_string();
        if field.attributes.secret {
            quote!(.field(#label, &#private::Redacted(&self.#ident)))
        } else {
            quote!(.field(#label, &self.#ident))
        }
    });
    let or_fields = fields.iter().map(|field| {
        let ident = &field.ident;
        quote!(#ident: self.#ident.or(lower.#ident))
    });
    let env_fields = fields.iter().map(|field| {
        let ident = &field.ident;
        let attributes = &field.attributes;
        let keep_blank = attributes.keep_blank;
        if let Some(from_env) = &attributes.from_env {
            return quote!(#ident: #from_env(env)?);
        }
        if attributes.env.is_empty() {
            return quote!(#ident: ::core::option::Option::None);
        }
        let names = &attributes.env;
        let read = attributes
            .read
            .clone()
            .map_or_else(|| quote!(#layered::read::parsed), |read| quote!(#read));
        quote!(#ident: #layered::read_env(env, &[#(#names),*], #keep_blank, #read)?)
    });
    let param_fields = fields.iter().map(|field| {
        let LayeredField { ident, ty, .. } = field;
        let attributes = &field.attributes;
        let keep_blank = attributes.keep_blank;
        if attributes.litellm_param.is_empty() {
            return quote!(#ident: ::core::option::Option::None);
        }
        let names = &attributes.litellm_param;
        let decode = match &attributes.deserialize_with {
            Some(path) => quote!(|value| #path(value).ok().flatten()),
            None => quote!(|value| <#ty as #private::serde::Deserialize>::deserialize(value).ok()),
        };
        quote!(#ident: #private::from_litellm_params(params, &[#(#names),*], #keep_blank, #decode))
    });

    let sources = fields.iter().map(|field| {
        let attributes = &field.attributes;
        let setting = attributes.setting.clone().unwrap_or_else(|| {
            LitStr::new(
                &field.ident.to_string().replace('_', " "),
                Span::call_site(),
            )
        });
        let params = &attributes.litellm_param;
        let env = &attributes.env;
        let litellm_setting = match &attributes.litellm_setting {
            Some(name) => quote!(::core::option::Option::Some(#name)),
            None => quote!(::core::option::Option::None),
        };
        quote!(#layered::Sources {
            setting: #setting,
            litellm_params: &[#(#params),*],
            litellm_setting: #litellm_setting,
            env: &[#(#env),*],
        })
    });
    let source_count = fields.len();
    let source_refs = fields.iter().enumerate().map(|(index, field)| {
        let constant = Ident::new(&field.ident.to_string().to_uppercase(), field.ident.span());
        quote!(pub const #constant: &'static #layered::Sources = &Self::SOURCES[#index];)
    });
    let litellm_params = fields
        .iter()
        .flat_map(|field| field.attributes.litellm_param.iter());
    let litellm_settings = fields.iter().filter_map(|field| {
        let setting = field.attributes.litellm_setting.as_ref()?;
        let label = field.ident.to_string();
        Some(quote!((#setting, #label)))
    });
    let env_names = unique(
        fields
            .iter()
            .flat_map(|field| field.attributes.env.iter().cloned()),
    );

    Ok(quote! {
        #[derive(::core::clone::Clone, ::core::default::Default, ::core::cmp::PartialEq)]
        #vis struct #partial {
            #(#partial_fields,)*
        }

        impl ::core::fmt::Debug for #partial {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                formatter.debug_struct(stringify!(#partial))
                    #(#debug_fields)*
                    .finish()
            }
        }

        impl #layered::Partial for #partial {
            fn or(self, lower: Self) -> Self {
                Self { #(#or_fields,)* }
            }
        }

        impl #partial {
            pub fn from_env(
                env: &(impl #layered::Lookup + ?::core::marker::Sized),
            ) -> ::core::result::Result<Self, #layered::Error> {
                ::core::result::Result::Ok(Self { #(#env_fields,)* })
            }

            pub fn from_litellm_params(
                params: &#private::serde_json::Map<::std::string::String, #private::serde_json::Value>,
            ) -> Self {
                Self { #(#param_fields,)* }
            }
        }

        impl #name {
            pub const SOURCES: [#layered::Sources; #source_count] = [#(#sources),*];
            pub const LITELLM_PARAMS: &'static [&'static str] = &[#(#litellm_params),*];
            pub const LITELLM_SETTINGS: &'static [(&'static str, &'static str)] = &[#(#litellm_settings),*];
            pub const ENV_NAMES: &'static [&'static str] = &[#(#env_names),*];
            #(#source_refs)*
        }
    })
}

fn unique(names: impl Iterator<Item = LitStr>) -> Vec<LitStr> {
    names.fold(Vec::new(), |mut seen, name| {
        if !seen
            .iter()
            .any(|known: &LitStr| known.value() == name.value())
        {
            seen.push(name);
        }
        seen
    })
}

fn field_attributes(field: &syn::Field) -> syn::Result<FieldAttributes> {
    let mut attributes = FieldAttributes::default();
    for attribute in field
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("layer"))
    {
        attribute.parse_nested_meta(|meta| {
            let path = &meta.path;
            if path.is_ident("env") {
                attributes.env = names(&meta)?;
            } else if path.is_ident("litellm_param") {
                attributes.litellm_param = names(&meta)?;
            } else if path.is_ident("read") {
                attributes.read = Some(meta.value()?.parse()?);
            } else if path.is_ident("from_env") {
                attributes.from_env = Some(meta.value()?.parse()?);
            } else if path.is_ident("deserialize_with") {
                attributes.deserialize_with = Some(meta.value()?.parse()?);
            } else if path.is_ident("litellm_setting") {
                attributes.litellm_setting = Some(meta.value()?.parse()?);
            } else if path.is_ident("setting") {
                attributes.setting = Some(meta.value()?.parse()?);
            } else if path.is_ident("secret") {
                attributes.secret = true;
            } else if path.is_ident("keep_blank") {
                attributes.keep_blank = true;
            } else {
                return Err(meta.error("unknown `layer` attribute"));
            }
            Ok(())
        })?;
    }
    if attributes.from_env.is_some() && (!attributes.env.is_empty() || attributes.read.is_some()) {
        return Err(syn::Error::new_spanned(
            field,
            "`from_env` reads the environment itself, so it cannot be combined with `env` or `read`",
        ));
    }
    if attributes.read.is_some() && attributes.env.is_empty() {
        return Err(syn::Error::new_spanned(field, "`read` needs `env`"));
    }
    if attributes.deserialize_with.is_some() && attributes.litellm_param.is_empty() {
        return Err(syn::Error::new_spanned(
            field,
            "`deserialize_with` needs `litellm_param`",
        ));
    }
    Ok(attributes)
}

fn names(meta: &ParseNestedMeta<'_>) -> syn::Result<Vec<LitStr>> {
    match meta.value()?.parse::<Expr>()? {
        Expr::Lit(syn::ExprLit {
            lit: Lit::Str(name),
            ..
        }) => Ok(vec![name]),
        Expr::Array(ExprArray { elems, .. }) => elems
            .into_iter()
            .map(|element| match element {
                Expr::Lit(syn::ExprLit {
                    lit: Lit::Str(name),
                    ..
                }) => Ok(name),
                other => Err(syn::Error::new_spanned(other, "expected a string literal")),
            })
            .collect(),
        other => Err(syn::Error::new_spanned(
            other,
            "expected a string literal or an array of them",
        )),
    }
}
